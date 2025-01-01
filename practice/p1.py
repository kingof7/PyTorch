# Import related library
import numpy as np
import torch

# Initializing a tensor
# 초기화 방법 1: 지정된 값으로 Tensor 초기화
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

print(x_data)
print(x_data.dtype)

# 초기화 방법 2: 랜덤한 값으로 Tensor 초기화
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# 초기화 방법 3: Numpy 배열로부터 Tensor 초기화
data = [[1, 2],[3, 4]]
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

print(x_np)
print(x_np.dtype)

# Tensor 데이터의 형식 지정
# Tensor의 데이터 형식은 어떤 것들이 있을까요?
# 1. 정수 int (integer) \
#     1-1. torch.int8, torch.int16, torch.int32, torch.int64 (torch.long) \
#     1-2. torch.uint8: unsigned integer로 양의 정수 0 ~ 255 숫자만 포함, 주로 이미지 데이터를 다룰 때 사용.
# 2. float \
#     2-1. torch.float16, torch.float32, torch.float64
# 3. boolean: torch.bool
# 4. etc.

# torch.float32 형식으로 초기화
data = [[1,2], [3,4]]
x_data = torch.tensor(data, dtype=torch.float32)
print(x_data)
print(x_data.dtype)

# torch.uint8 형식으로 초기화
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data, dtype=torch.uint8)
print(x_data)
print(x_data.dtype)

# 다른 dtype으로 변환
x_data = x_data.to(torch.float16)
print(x_data.dtype)

# 다른 Device로 Tensor 옮기기

# 1. GPU가 사용가능한지 확인하기
# 2. 다른 Device로 Tensor 옮기기
# 3. 어떤 Device상에 Tensor가 있는지 확인하기

# GPU가 사용가능한지 확인하기
print(torch.cuda.is_available())

# GPU로 Tensor 옮기기
# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

# CPU로 Tensor 옮기기
x_data = x_data.cpu()
# x_data = x_data.to("cpu")

# 어떤 Device상에 Tensor가 있는지 확인하기
print(x_data.device)

# Tensor을 활용한 연산
# 1. Indexing and slicing
# 2. Concatenation
# 3. Arithmetric
# 4. Inplace-operation

# Indexing
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0 # 모든 행에대한 index 1에 해당하는 값을 0으로 바꿈
print(tensor)

# Concatenation
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# Arithmetric
y1 = tensor @ tensor.T # 행렬의 곱
y2 = tensor.matmul(tensor.T) # Matrix Multiple
print(y1)
print(y2)

# Element-wise 곱셈
z1 = tensor * tensor
z2 = tensor.mul(tensor)
print(z1)
print(z2)

# Inplace operation
print(f"{tensor} \n")
tensor.add_(5) # 각 원소에 5씩 더해주고 override(=Inplace) 해줌
print(tensor)