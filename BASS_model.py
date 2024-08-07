"""

Code taken from (with some modification) https://github.com/fbickfordsmith/epig/tree/b11124d2dd48381a5756e14d920d401f1fd3120d

The authors of the paper use a "trainer" and a "model" that work together to do their BASS framework

As of 8/6/2024 at 1:49pm, this is transfered except for laplace bit!

"""
import math
from typing import Any, Tuple, Union, Callable, Sequence
from dataclasses import dataclass
from operator import gt, lt
from time import time

import torch
from torch import Tensor, Generator
from torch.nn import Dropout, Linear, Module, ReLU, Sequential
from torch.func import functional_call, grad, vmap
from torch.nn.functional import log_softmax, nll_loss, softmax
from torch.nn.utils import vector_to_parameters
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from tqdm import tqdm
import numpy as np
# need to figure this stuff out
from laplace import DiagLaplace, DiagSubnetLaplace, ParametricLaplace
from laplace.utils import (
    LargestMagnitudeSubnetMask,
    LastLayerSubnetMask,
    ParamNameSubnetMask,
    RandomSubnetMask,
    SubnetMask,
)

# can possibly change this to import *
from BASS_utils import (
    accuracy_from_marginals,
    nll_loss_from_probs,
    count_correct_from_marginals,
    get_next,
    Dictionary,
    prepend_to_keys
)

from BASS_uncertainty import (
    epig_from_probs,
    epig_from_probs_using_matmul,
    epig_from_probs_using_weights
)

import globals

# Model Portion

# taken from ./src/models/fc_net.py
class FullyConnectedNet(Module):
    def __init__(
        self,
        input_shape: Sequence[int],
        hidden_sizes: Sequence[int],
        output_size: int,
        activation_fn: Callable = ReLU,
        dropout_rate: float = 0.0,
        use_input_dropout: bool = False,
    ) -> None:
        super().__init__()
        
        sizes = (math.prod(input_shape), *hidden_sizes)
        layers = []

        if (dropout_rate > 0) and use_input_dropout:
            layers += [Dropout(p=dropout_rate)]

        for i in range(len(sizes) - 1):
            layers += [Linear(in_features=sizes[i], out_features=sizes[i + 1])]
            layers += [activation_fn()]

            if (dropout_rate > 0) and (i < len(sizes) - 2):
                layers += [Dropout(p=dropout_rate)]

        layers += [Linear(in_features=sizes[-1], out_features=output_size)]

        self.layers = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor[float], [N, *F]

        Returns:
            Tensor[float], [N, O]
        """
        x = x.flatten(start_dim=1).to(globals.device)  # [N, F]
        return self.layers(x)  # [N, O]
    

# Trainer portion

# taken from ./src/trainers/base.py

class Trainer:
    def test(self, loader: DataLoader, n_classes: int = None) -> dict:
        """
        loss = 1/N ∑_{i=1}^N L(x_i,y_i)

        Here we use
            L_1(x_i,y_i) = bin_loss(x_i,y_i) = 1[argmax p(y|x_i) == y_i]
            L_2(x_i,y_i) = mae_loss(x_i,y_i) = |E[p(y|x_i)] - y_i|
            L_3(x_i,y_i) = mse_loss(x_i,y_i) = (E[p(y|x_i)] - y_i)^2
            L_4(x_i,y_i) = nll_loss(x_i,y_i) = -log p(y_i|x_i).

        For stochastic models we use
            p(y|x_i)  = E_{p(θ)}[p(y|x_i,θ)]
                     ~= 1/K ∑_{j=1}^K p(y|x_i,θ_j), θ_j ~ p(θ).
        """
        self.eval_mode()

        test_log = Dictionary() # why do they re-implement a Dictionary
        n_examples = 0

        for inputs, labels in loader:
            inputs = inputs.to(globals.device)
            labels = labels.to(globals.device)
            if n_classes is not None:
                test_log_update = self.evaluate_test(inputs, labels, n_classes)
            else:
                test_log_update = self.evaluate_test(inputs, labels)

            test_log.append(test_log_update)

            n_examples += len(inputs)

        test_log = test_log.concatenate()

        for metric, scores in test_log.items():
            test_log[metric] = torch.sum(scores).item() / n_examples

        if "n_correct" in test_log:
            test_log["acc"] = test_log.pop("n_correct")

        return test_log
    
    
class StochasticTrainer(Trainer):
    """
    Base trainer for a stochastic model.
    """

    def estimate_uncertainty(
        self, loader: DataLoader, method: str, seed: int, inputs_targ: Tensor = None
    ) -> Tensor:
        self.eval_mode()

        estimate_epig_using_pool = (
            (method == "epig")
            and hasattr(self.epig_cfg, "target_class_dist")
            and self.epig_cfg.target_class_dist is not None
        )

        if estimate_epig_using_pool:
            scores = self.estimate_epig_using_pool(loader, n_input_samples=len(inputs_targ))  # [N,]

        else:
            scores = []

            for inputs_i, _ in loader:
                self.set_rng_seed(seed)

                if method == "epig":
                    scores_i = self.estimate_epig_batch(inputs_i, inputs_targ)  # [B,]
                else:
                    scores_i = self.estimate_uncertainty_batch(inputs_i, method)  # [B,]

                scores += [scores_i.cpu()]

            scores = torch.cat(scores)  # [N,]

        return scores  # [N,]


# taken from ./src/trainers/pytorch.py

@dataclass
class PyTorchTrainer:
    model: Module
    optimizer: Optimizer
    torch_rng: Generator
    n_optim_steps_min: int
    n_optim_steps_max: int
    n_samples_train: int
    n_samples_test: int
    n_validations: int
    early_stopping_metric: str
    early_stopping_patience: int
    restore_best_model: bool
    epig_cfg: dict = None

    def __post_init__(self) -> None:
        self.optimizer = self.optimizer(params=self.model.parameters(),lr=0.01,momentum=0.95,weight_decay=0) # defaults from github repo
        self.validation_gap = max(1, int(self.n_optim_steps_max / self.n_validations))

    def set_rng_seed(self, seed: int) -> None:
        self.torch_rng.manual_seed(seed)

    def train(
        self, train_loader: DataLoader, val_loader: DataLoader, verbose: bool = False
    ) -> Tuple[int, Dictionary]:
        log = Dictionary()
        start_time = time()

        step_range = range(self.n_optim_steps_max)
        step_range = tqdm(step_range, desc="Training") if verbose else step_range

        best_score = 0 if "acc" in self.early_stopping_metric else math.inf
        early_stopping_operator = gt if "acc" in self.early_stopping_metric else lt

        for step in step_range:
            train_metrics = self.train_step(train_loader)

            if step % self.validation_gap == 0:
                with torch.inference_mode():
                    val_metrics = self.test(val_loader)

                log_update = {
                    "time": time() - start_time,
                    "step": step,
                    **prepend_to_keys(train_metrics, "train"),
                    **prepend_to_keys(val_metrics, "val"),
                }
                log.append(log_update)

                latest_score = log_update[self.early_stopping_metric]
                score_has_improved = early_stopping_operator(latest_score, best_score)

                if (step < self.n_optim_steps_min) or score_has_improved:
                    best_model_state = self.model.state_dict()
                    best_score = latest_score
                    patience_left = self.early_stopping_patience
                else:
                    patience_left -= self.validation_gap

                if (self.early_stopping_patience != -1) and (patience_left <= 0):
                    break

        if self.restore_best_model:
            self.model.load_state_dict(best_model_state)

        self.postprocess_model(train_loader)

        return step, log

    def postprocess_model(self, train_loader: DataLoader) -> None:
        pass

# taken from ./src/trainers/pytorch_classif.py

class PyTorchClassificationTrainer(PyTorchTrainer):
    def train_step(self, loader: DataLoader) -> dict:
        inputs, labels = get_next(loader)  # [N, ...], [N,]

        self.model.train()
        breakpoint()
        acc, nll = self.evaluate_train(inputs, labels)  # [1,], [1,]

        self.optimizer.zero_grad()
        nll.backward()
        self.optimizer.step()

        return {"acc": acc.item(), "nll": nll.item()}

    def split_params(self, embedding_params: Sequence[str]) -> Tuple[dict, dict]:
        model = self.model.model if isinstance(self.model, ParametricLaplace) else self.model

        grad_params, no_grad_params = {}, {}

        for name, param in model.named_parameters():
            if name in embedding_params:
                grad_params[name] = param.detach()
            else:
                no_grad_params[name] = param.detach()

        return grad_params, no_grad_params

    def compute_badge_embeddings_v1(
        self, loader: DataLoader, embedding_params: Sequence[str]
    ) -> Tensor:
        self.eval_mode()

        model = self.model.model if isinstance(self.model, ParametricLaplace) else self.model

        embeddings = []

        for inputs, _ in loader:
            pseudolosses = self.compute_badge_pseudoloss_v1(inputs)  # [B,]

            for pseudoloss in pseudolosses:
                # Prevent the grad attribute of each tensor accumulating a sum of gradients.
                model.zero_grad()

                pseudoloss.backward(retain_graph=True)

                embedding_i = []

                for name, param in model.named_parameters():
                    if name in embedding_params:
                        gradient = param.grad.detach().flatten().cpu()  # [E',]
                        embedding_i += [gradient]

                embedding_i = torch.cat(embedding_i)  # [E,]
                embeddings += [embedding_i]  # [E,]

        return torch.stack(embeddings)  # [N, E]

    def compute_badge_embeddings_v2(
        self, loader: DataLoader, embedding_params: Sequence[str]
    ) -> Tensor:
        self.eval_mode()

        grad_params, no_grad_params = self.split_params(embedding_params)

        compute_grad = grad(self.compute_badge_pseudoloss_v2, argnums=1)
        compute_grad = vmap(compute_grad, in_dims=(0, None, None), randomness="same")

        embeddings = []

        for inputs, _ in loader:
            gradient_dict = compute_grad(inputs, grad_params, no_grad_params)

            gradients = []

            for name, gradient in gradient_dict.items():
                gradient = gradient.flatten(start_dim=1).cpu()
                gradients += [gradient]

            gradients = torch.cat(gradients, dim=-1)
            embeddings += [gradients]

        return torch.cat(embeddings)  # [N, E]
    





# taken from ./src/trainers/base_classif_probs.py
class ProbsClassificationStochasticTrainer(StochasticTrainer):
    """
    Base trainer for a stochastic classification model that outputs the probs of a categorical
    predictive distribution.
    """

    """
    uncertainty_estimators = {
        "bald": bald_from_probs,
        "marg_entropy": marginal_entropy_from_probs,
        "mean_std": mean_standard_deviation_from_probs,
        "pred_margin": predictive_margin_from_probs,
        "var_ratio": variation_ratio_from_probs,
    }
    """
    def evaluate_test(self, inputs: Tensor, labels: Tensor, n_classes: int = None) -> dict:
        probs = self.marginal_predict(inputs, self.n_samples_test)  # [N, Cl]

        if (n_classes is not None) and (n_classes < probs.shape[-1]):
            probs = probs[:, :n_classes]  # [N, n_classes]
            probs /= torch.sum(probs, dim=-1, keepdim=True)  # [N, n_classes]

        n_correct = count_correct_from_marginals(probs, labels)  # [1,]
        nll = nll_loss_from_probs(probs, labels, reduction="sum")  # [1,]

        return {"n_correct": n_correct, "nll": nll}

    def estimate_uncertainty_batch(self, inputs: Tensor, method: str) -> Tensor:
        uncertainty_estimator = self.uncertainty_estimators[method]

        probs = self.conditional_predict(
            inputs, self.n_samples_test, independent=True
        )  # [N, K, Cl]

        return uncertainty_estimator(probs)  # [N,]

    def estimate_epig_batch(self, inputs_pool: Tensor, inputs_targ: Tensor) -> Tensor:
        probs = self.conditional_predict(
            torch.cat((inputs_pool, inputs_targ)), self.n_samples_test, independent=False
        )  # [N_p + N_t, K, Cl]

        probs_pool = probs[: len(inputs_pool)]  # [N_p, K, Cl]
        probs_targ = probs[len(inputs_pool) :]  # [N_t, K, Cl]

        if self.epig_cfg.use_matmul:
            scores = epig_from_probs_using_matmul(probs_pool, probs_targ)  # [N_p,]
        else:
            scores = epig_from_probs(probs_pool, probs_targ)  # [N_p,]

        return scores  # [N_p,]

    def estimate_epig_using_pool(self, loader: DataLoader, n_input_samples: int = None) -> Tensor:
        probs_cond = []

        for inputs, _ in loader:
            probs_cond_i = self.conditional_predict(
                inputs, self.n_samples_test, independent=True
            )  # [B, K, Cl]
            probs_cond += [probs_cond_i]

        probs_cond = torch.cat(probs_cond)  # [N, K, Cl]
        probs_marg = torch.mean(probs_cond, dim=1)  # [N, Cl]
        probs_marg_marg = torch.mean(probs_marg, dim=0, keepdim=True)  # [1, Cl]

        # Compute the weights, w(x_*) ~= ∑_{y_*} p_*(y_*) p_{pool}(y_*|x_*) / p_{pool}(y_*).
        target_class_dist = self.epig_cfg.target_class_dist
        target_class_dist = torch.tensor([target_class_dist]).to(inputs.device)  # [1, Cl]
        target_class_dist /= torch.sum(target_class_dist)  # [1, Cl]
        weights = torch.sum(target_class_dist * probs_marg / probs_marg_marg, dim=-1)  # [N,]

        # Ensure that ∑_{x_*} w(x_*) == N.
        assert math.isclose(torch.sum(weights).item(), len(weights), rel_tol=1e-3)

        # Compute the weighted EPIG scores.
        scores = []

        if n_input_samples is not None:
            # We do not need to normalize the weights before passing them to torch.multinomial().
            inds = torch.multinomial(
                weights, num_samples=n_input_samples, replacement=True
            )  # [N_s,]

            probs_targ = probs_cond[inds]  # [N_s, K, Cl]

            for probs_cond_i in torch.split(probs_cond, len(inputs)):
                if self.epig_cfg.use_matmul:
                    scores_i = epig_from_probs_using_matmul(probs_cond_i, probs_targ)  # [B,]
                else:
                    scores_i = epig_from_probs(probs_cond_i, probs_targ)  # [B,]

                scores += [scores_i.cpu()]

        else:
            probs_targ = probs_cond  # [N, K, Cl]

            for probs_cond_i in torch.split(probs_cond, len(inputs)):
                scores_i = epig_from_probs_using_weights(probs_cond_i, probs_targ, weights)  # [B,]
                scores += [scores_i.cpu()]

        return torch.cat(scores)  # [N,]

# taken from ./src/trainers/pytorch_classif_laplace_approx.py

class PyTorchClassificationLaplaceTrainer(
    PyTorchClassificationTrainer, ProbsClassificationStochasticTrainer
):
    def __init__(
        self,
        laplace_approx: ParametricLaplace,
        likelihood_temperature: Union[float, int, str] = 1,
        subnet_mask: SubnetMask = None,
        subnet_mask_inds: Sequence[int] = None,
        subnet_mask_names: Sequence[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.laplace_approx = laplace_approx
        self.likelihood_temperature = likelihood_temperature
        self.subnet_mask = subnet_mask
        self.subnet_mask_inds = subnet_mask_inds
        self.subnet_mask_names = subnet_mask_names

    def eval_mode(self) -> None:
        if isinstance(self.model, ParametricLaplace):
            self.model.model.eval()
        else:
            self.model.eval()

    def sample_model_parameters(self, n_model_samples: int, sample_on_cpu: bool = False) -> Tensor:
        """
        Sample from the parameter distribution with control over the random-number generator, which
        is not possible using the sample() method of DiagLaplace or DiagSubnetLaplace. If we want to
        handle self.model being another subclass of ParametricLaplace, we need to adapt sample()
        from that subclass.
        """
        assert isinstance(self.model, DiagLaplace)

        if isinstance(self.model, DiagSubnetLaplace):
            n_params = self.model.n_params_subnet
        else:
            n_params = self.model.n_params

        if sample_on_cpu:
            device = self.torch_rng.device

            seed = torch.randint(high=int(1e6), size=[1], generator=self.torch_rng, device=device)
            seed = seed.item()

            torch_rng_cpu = torch.Generator().manual_seed(seed)

            mean = self.model.mean.new_zeros(n_model_samples, n_params, device="cpu")  # [K, P]
            std = self.model.mean.new_ones(n_model_samples, n_params, device="cpu")  # [K, P]
            samples = torch.normal(mean, std, generator=torch_rng_cpu).to(device)  # [K, P]

        else:
            mean = self.model.mean.new_zeros(n_model_samples, n_params)  # [K, P]
            std = self.model.mean.new_ones(n_model_samples, n_params)  # [K, P]
            samples = torch.normal(mean, std, generator=self.torch_rng)  # [K, P]

        samples *= self.model.posterior_scale[None, :]  # [K, P]

        if isinstance(self.model, DiagSubnetLaplace):
            samples += self.model.mean_subnet[None, :]
            return self.model.assemble_full_samples(samples)
        else:
            samples += self.model.mean[None, :]
            return samples

    def conditional_predict(
        self, inputs: Tensor, n_model_samples: int, independent: bool
    ) -> Tensor:
        """
        The predictive_samples() method of ParametricLaplace takes a pred_type argument.

        pred_type="glm":
        - Idea: compute a Gaussian over latent-function values, sample from the Gaussian, then pass
          the sampled latent-function values through a softmax.
        - Issue: involves calling _glm_predictive_distribution(), which in turn involves computing
          a Jacobian matrix, which uses a lot of memory if the number of classes is big.

        pred_type="nn":
        - Idea: sample from the Gaussian over the model parameters, then for each sampled parameter
          configuration compute a forward pass through the model.
        - Issue: involves calling _nn_predictive_samples(), which does not allow for passing a
          random-number generator (needed to ensure samples are the same across data batches).

        We use pred_type="nn" and address the issue by reimplementing _nn_predictive_samples()
        within this function.

        References:
            https://github.com/aleximmer/Laplace/blob/main/laplace/baselaplace.py#L684

        Arguments:
            inputs: Tensor[float], [N, *F]
            n_model_samples: int
            independent: bool

        Returns:
            Tensor[float], [N, K, Cl]
        """
        assert isinstance(self.model, ParametricLaplace)

        probs = []

        for param_sample in self.sample_model_parameters(n_model_samples):
            # Set the model parameters to the sampled configuration.
            vector_to_parameters(param_sample, self.model.model.parameters())

            features = self.model.model(inputs)  # [N, Cl]
            probs += [softmax(features, dim=-1)]

        # Set the model parameters to the mean.
        vector_to_parameters(self.model.mean, self.model.model.parameters())

        return torch.stack(probs, dim=1)  # [N, K, Cl]

    def marginal_predict(self, inputs: Tensor, n_model_samples: int) -> Tensor:
        """
        Arguments:
            inputs: Tensor[float], [N, *F]
            n_model_samples: int

        Returns:
            Tensor[float], [N, Cl]
        """
        if isinstance(self.model, ParametricLaplace):
            probs = self.conditional_predict(
                inputs, n_model_samples, independent=True
            )  # [N, K, Cl]
            return torch.mean(probs, dim=1)  # [N, Cl]

        else:
            features = self.model(inputs)  # [N, Cl]
            return softmax(features, dim=-1)  # [N, Cl]

    def evaluate_train(self, inputs: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        """
        loss = 1/N ∑_{i=1}^N L(x_i,y_i,θ)

        Here we use
            L_1(x_i,y_i,θ) = nll_loss(x_i,y_i,θ) = -log p(y_i|x_i,θ)
            L_2(x_i,y_i,θ) = binary_loss(x_i,y_i,θ) = {argmax p(y|x_i,θ) != y_i}
        """
        # got an error with inputs and weights being on different devices
        # so we are moving inputs to globals.device
        inputs = inputs.to(globals.device)
        labels = labels.to(globals.device)
        features = self.model(inputs)  # [N, Cl]
        logprobs = log_softmax(features, dim=-1)  # [N, Cl]

        acc = accuracy_from_marginals(logprobs, labels)  # [1,]
        nll = nll_loss(logprobs, labels)  # [1,]

        return acc, nll  # [1,], [1,]

    def postprocess_model(self, train_loader: DataLoader) -> None:
        if self.subnet_mask is not None:
            simple_mask_fns = (LargestMagnitudeSubnetMask, LastLayerSubnetMask, RandomSubnetMask)

            if self.subnet_mask.func in simple_mask_fns:
                subnet_mask = self.subnet_mask(self.model)

            elif self.subnet_mask.func == ParamNameSubnetMask:
                param_names = []
                param_count = 0
                param_inds = {n: 0 for n in self.subnet_mask_names}

                for name, param in self.model.named_parameters():
                    for _name in self.subnet_mask_names:
                        if (_name in name) and (param_inds[_name] in self.subnet_mask_inds):
                            param_names += [name]
                            param_count += param.numel()
                            param_inds[_name] += 1

                subnet_mask = self.subnet_mask(self.model, param_names)

            else:
                raise ValueError

            laplace_approx_kwargs = dict(subnetwork_indices=subnet_mask.select())

        else:
            param_count = sum(param.numel() for param in self.model.parameters())

            laplace_approx_kwargs = {}

        if self.likelihood_temperature == "inverse_param_count":
            laplace_approx_kwargs["temperature"] = 1 / param_count
        elif isinstance(self.likelihood_temperature, (float, int)):
            laplace_approx_kwargs["temperature"] = self.likelihood_temperature
        else:
            raise NotImplementedError

        self.model = self.laplace_approx(self.model, **laplace_approx_kwargs)
        self.model.fit(train_loader)
        self.model.optimize_prior_precision(method="marglik")

    def compute_badge_pseudoloss_v1(self, inputs: Tensor) -> Tensor:
        """
        Arguments:
            inputs: Tensor[float], [N, *F]

        Returns:
            Tensor[float], [N,]
        """
        assert isinstance(self.model, ParametricLaplace)

        features = self.model.model(inputs)  # [N, Cl]
        logprobs = log_softmax(features, dim=-1)  # [N, Cl]
        pseudolabels = torch.argmax(logprobs, dim=-1)  # [N,]

        return nll_loss(logprobs, pseudolabels, reduction="none")  # [N,]

    def compute_badge_pseudoloss_v2(
        self, _input: Tensor, grad_params: dict, no_grad_params: dict
    ) -> Tensor:
        """
        Arguments:
            inputs: Tensor[float], [1, *F]

        Returns:
            Tensor[float], [1,]
        """
        features = functional_call(
            self.model, (grad_params, no_grad_params), _input[None, :]
        )  # [1, Cl]

        logprobs = log_softmax(features, dim=-1)  # [1, Cl]
        pseudolabel = torch.argmax(logprobs, dim=-1)  # [1,]

        return nll_loss(logprobs, pseudolabel)  # [1,]




