import torch

# Define parameters
batch_size = 32  # mini-batch size
feature_size = 128  # feature dimension

# Create a random tensor
x = torch.rand(batch_size, feature_size)

# Print tensor information
print(f"Tensor shape: {x.shape}")
print(f"Tensor device: {x.device}")
print(f"Tensor dtype: {x.dtype}")

# Example operations with the tensor
print("\nSome basic tensor operations:")
print(f"Mean across batch: {x.mean(dim=0).shape}")  # Shape: [feature_size]
print(f"Sum across features: {x.sum(dim=1).shape}")  # Shape: [batch_size]
print(f"Max value: {x.max().item():.4f}")
print(f"Min value: {x.min().item():.4f}")
