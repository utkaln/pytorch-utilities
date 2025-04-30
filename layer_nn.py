import torch
import torch.nn as nn

input_tensor = torch.Tensor([[2,3,4,5,6,7,8]])
nn_model = nn.Sequential(
    nn.Linear(7,4),
    nn.Linear(4,2),
)

# Calculate total parameters
# parameters per neuron = count of input neurons + 1 (bias)
total_parameters = 0
for param in nn_model.parameters():
    total_parameters += param.numel()
print(f"Total parameters: {total_parameters}")


output_tensor = nn_model(input_tensor)
print(output_tensor)
print(output_tensor.shape)