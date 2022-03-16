5.1
There are three methods to define modules based on nn.Module: Sequential, ModuleList and ModuleDict
Sequential(nn.Sequential) can use an ordered dictionary(OrderedDict). The definition of Sequential has _init_ and forward function, and they provide the order of input.
```
#normal
import torch.nn as nn
net = nn.Sequential(
        nn.Linear(784, 256),#input, output neurons
        nn.ReLU(),
        nn.Linear(256, 10), 
        )
print(net)

#OrderedDict
import collections
import torch.nn as nn
net2 = nn.Sequential(collections.OrderedDict([
          ('fc1', nn.Linear(784, 256)),
          ('relu1', nn.ReLU()),
          ('fc2', nn.Linear(256, 10))
          
          ]))
print(net2)
```

```
Sequential(
  (0): Linear(in_features=784, out_features=256, bias=True)
  (1): ReLU()
  (2): Linear(in_features=256, out_features=10, bias=True)
)
Sequential(
  (fc1): Linear(in_features=784, out_features=256, bias=True)
  (relu1): ReLU()
  (fc2): Linear(in_features=256, out_features=10, bias=True)
)
```
ModuleList(nn.ModuleList) can receive module as input, its operation likes List's. 
But ModuleList does not have order, it should use 'forward' function to set the order of modules it stored.
ModuleDict(nn.ModuleDict) likes ModuleList. And it can easier to add layer.

5.1
For complex module, it has lots of repeated structures. So, we can define these and add them when use(like functions in code).
U-Net:Double Convolution, Max pooling, Up sampling, output. (can use forward function to link these block)

5.3
Change module based on original module.
Change layer(fc): define a structure(classifier) like 5.1. Use classifier to replace original part(net.fc=classifier).


