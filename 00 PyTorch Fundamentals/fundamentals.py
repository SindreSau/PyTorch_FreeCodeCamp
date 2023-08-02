import torch

# SCALARS, VECTORS, MATRICES, AND TENSORS

# Scalars are typically stored in lowercase (e.g. a, b, c)
# Vectors are typically stored in lowercase with an arrow above (e.g. x, y, z)
# Matrices are typically stored in uppercase (e.g. Q, R, S)
# Tensors are typically stored in uppercase with an arrow above (e.g. X, Y, Z)

# scalar
scalar = torch.tensor(7)
print(scalar)
print("dimension:", scalar.ndim)
print("shape:", scalar.shape)
print("item:", scalar.item())
print("")

# vector
vector = torch.tensor([4, 2])
print(vector)
print("dimension:", vector.ndim)
print("shape:", vector.shape)
print("")

# matrix
MATRIX = torch.tensor([[1, 2], [4, 5]])
print(MATRIX)
print("dimension:", MATRIX.ndim)
print("shape:", MATRIX.shape)
print("")

# tensor
TENSOR = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9], ]])
print(TENSOR)
print("dimension:", TENSOR.ndim)
print("shape:", TENSOR.shape)
print("")

# RANDOM TENSORS
# Random tensors are important because the way many neural networks learn, is that they start
# with tensors full of random numbers and then adjust those numbers to better represent the data

# Start with random numbers -> look at data -> update random numbers _-> look at data -> update random numbers -> etc.

# Create a random tensor of size (3, 4)
random_tensor = torch.rand(3, 4)
print(random_tensor)
print("")

# Typical image tensor
# Color channels(RGB), height, width
random_image_tensor = torch.rand(3, 28, 28)  # 3 color channels, 28x28 pixels
print("Image tensor:", random_image_tensor)
print("Image tensor shape:", random_image_tensor.shape)
print("Image tensor dimension:", random_image_tensor.ndim)
print("")

