import numpy
import torch
import torch_teachers

def getPredictedLabels(data, aggregator, num_models=250):
    votes = [] # final voting record
    for i in range(num_models):
        state_dict = torch.load('./saved/teacher_'+str(i)+'.txt')
        m = torch_teachers.CNN()
        m.load_state_dict(state_dict)
        m.eval()

        ballot = [] # user i's voting record

        for p in data:
            ballot.append(m(p))
        votes.append(ballot)
    return aggregator(torch.transpose(torch.Tensor(votes),0,1))
