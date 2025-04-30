import torch.nn as nn
import torch

input_tensor = torch.tensor([[0.3471, 0.4547, -0.2356]])
linear_layer = nn.Linear(in_features=3, out_features=1)
output_tensor = linear_layer(input_tensor)
print(output_tensor)
print(output_tensor.shape)