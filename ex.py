# Let's create the tensors as described and compute the pooled similarity matrix to verify the approach.

import torch

# Define dimensions
B, L, V, D = 10, 2, 3, 6  # Example dimensions

# Create tensors
tensor1 = torch.ones((B, L, D))  # Tensor filled with ones
tensor2 = torch.arange(B).view(B, 1, 1).repeat(1, V, D).float()  # Tensor with batch numbers repeated

tensor1 = tensor1.view(-1, D)
tensor2 = tensor2.view(-1, D)
# Compute similarity matrix using einsum (inner product)
similarity_matrix = torch.einsum('id,jd->ij', tensor1, tensor2)  # Adjusted indices for clarity

# Reshape and permute the resulting similarity matrix to get (B, B, L, V) shape
similarity_matrix_reshaped = similarity_matrix.view(B, L, B, V).permute(0, 2, 1, 3)

# Perform mean pooling over L and V dimensions to get a (B, B) shape
pooled_similarity_matrix = similarity_matrix_reshaped.mean(dim=-1).mean(dim=-1)

print("Pooled Similarity Matrix Shape:", pooled_similarity_matrix.shape)
print(pooled_similarity_matrix)
