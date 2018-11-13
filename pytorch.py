from __future__ import print_function
import torch
import numpy as np

# Construct and unitiliazed 5x3 matrix:
x = torch.empty(5, 3)
print(x)

# Construct a randomly initialized matrix:
x = torch.rand(5, 3)
print(x)

# Construct a matrix filled zeros and of dtype long:
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# Create a tensor based on an existing tensor
x = x.new_ones(5, 3, dtype=torch.double)
print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)                                      # result has the same size

# Get size of tensor
print(x.size())

# Operations
y = torch.rand(5, 3)
print(x+y)
print(torch.add(x,y))

# Addition: providing an output tensor as argument
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# adds x to y
y.add_(x)
print(y)

# Note: Any operation that mutates a tensor in-place is post-fixed with an _. 
# For example: x.copy_(y), x.t_(), will change x.

# Indexing
print(x[:, 1]) #gives 2nd column, ':' for all elements

# To resize/reshape tensor, use torch.view
x = torch.rand(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

# For one element tensor use item() to get value as a Python number
x = torch.randn(1)
print(x)
print(x.item())

# Converting a Torch Tensor to a NumPy array and vice versa is easy.
# The Torch Tensor and NumPy array will share their underlying memory locations, 
# and changing one will change the other.
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

# See how numpy array (b) changes its value:
a.add_(1)
print(a)
print(b)

# Going the other way:
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
