should we pass new "dataset object" to aggregator? Uses:
- num labels

In general, what should we do about parameters defaulting to "svhn" and svhn-associated values?
- option 1: pass just the dataset name and access the object
- option 2: pass the dataset object directly
- option 3: just leave it be, it doesn't need to be perfect
Resolution: stick with 3 until the end, then can change if need be

datset object params/attributes:
- dataset name
- num teachers
- randomizer, optional
- "split" name (teach_train, teach_valid, student_train, ...)
- label overriding
- model architecture: transforms, also maybe internal layers? (ask Carter)

torch `random_split` wants a pytorch generator:
- will it take a numpy generator?
- is there anything that needs a numpy generator?
(if possible, want to only have one kind of generator)

use underscores for anything not meant to be accessed by the user



**SCOPE CREEP**: `assert`ions that mess-up-able (default) params are valid
e.g. `assert model in ["teach", "student"]`
and generally cleaning up default paramters
**this is a separate problem, along the lines of above option 3
