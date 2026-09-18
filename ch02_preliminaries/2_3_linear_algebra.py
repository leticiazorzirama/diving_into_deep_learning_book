import torch

# 2.3.1. Scalars
# scalars = one number at a time

x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y

# 2.3.2. Vectors
# vector = fixed-length array of scalars
# we call these scalars the elements of the vector (synonyms include entries and components)

# Caution: in Python, as in most programming languages, vector indices start at, 
# also known as zero-based indexing, whereas in linear algebra subscripts begin at 
# (one-based indexing).

x = torch.arange(3)
print(x)
x[2]
len(x)

# The shape is a tuple that indicates a tensor’s length along each axis. Tensors with just one axis have shapes with just one element.
print(x.shape)

# Oftentimes, the word “dimension” gets overloaded to mean both the number of axes and the 
# length along a particular axis. To avoid this confusion, we use order to refer to the number 
# of axes and dimensionality exclusively to refer to the number of components.

# 2.3.3. Matrices
# scalars = 0-order tensors 
# vectors = 1-order tensors
# matrices = 2-order tensors

# matrix A contains m rows x n columns
# m = n is a square matrix
# i = m rows indices
# j = n columns indices

A = torch.arange(6).reshape(3, 2)
print(A)

# transpose
print(A.T)

# test transposes
# a i j 
A[0,1]
A[1,1]
A[2,0]

# a j i 
A.T[1,0]
A.T[1,1]
A.T[0,2]

# symmetric matrices
A = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(A)
print(A == A.T)

# 2.3.4. Tensors

# high-order tensors
torch.arange(24).reshape(2, 3, 4)

# 2.3.5. Basic Properties of Tensor Arithmetic

# element-wise operations
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = A.clone()  # Assign a copy of A to B by allocating new memory
A, A + B

# Hadamard product = elementwise product of two matrices
A * B

# Adding or multiplying a scalar and a tensor produces a result with the same shape as the original tensor
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape

# 2.3.6. Reduction

# sum of the elements in a vector
x = torch.arange(3, dtype=torch.float32)
x, x.sum()

# sum over the elements of tensors with arbitrary shape
A.shape, A.sum()

# sum over the axis = 0
print(A)
print(A.shape)
A.sum(axis=0)
print(A.sum(axis=0).shape)

# sum over the axis = 1
print(A)
print(A.shape)
A.sum(axis=1)
print(A.sum(axis=1).shape)

# reduce along both rows and columns
print(A.sum(axis=[0, 1]) == A.sum())  # Same as A.sum()

# mean
A.mean(), A.sum() / A.numel()

# calculating the mean along specific axes
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
A.mean(axis=1), A.sum(axis=1) / A.shape[1]

# 2.3.7. Non-Reduction Sum
# broadcast mechanism
sum_A = A.sum(axis=1, keepdims=True)
print(sum_A, sum_A.shape)

A / sum_A

# cumulative sum
A.cumsum(axis=0)

# 2.3.8. Dot Products

y = torch.ones(3, dtype = torch.float32)
x, y, torch.dot(x, y)

# dot product = inner product
# Given two vectors, their dot product is a sum over the products of the elements at the same position
# x * y > x.T * y

torch.sum(x * y)

# ...given some set of values, denoted by a vector x, and a set of weights, denoted by w, 
# the weighted sum of the values in X according to the weights could be expressed as the dot 
# product. When the weights are nonnegative and sum to 1, the dot product expresses a weighted 
# average. 
# After normalizing two vectors to have unit length, the dot products express the cosine of the 
# angle between them. Later in this section, we will formally introduce this notion of length.

# 2.3.9. Matrix–Vector Products
# Matrix–vector products also describe the key calculation involved in computing the outputs of 
# each layer in a neural network given the outputs from the previous layer.
A.shape, x.shape, torch.mv(A, x), A@x

# 2.3.10. Matrix–Matrix Multiplication
B = torch.ones(3, 4)
torch.mm(A, B), A@B

# 2.3.11. Norms
# l2 norm = euclidean lenght of a vector
# notion of size, not its dimensionality

# euclidean norm
u = torch.tensor([3.0, -4.0])
torch.norm(u)

# l1 norm = manhattan distance
torch.abs(u).sum()

# frobenius norm = behaves as if it were an l2 norm of a matrix-shaped vector
torch.norm(torch.ones((4, 9)))

# 2.3.13. Exercises
