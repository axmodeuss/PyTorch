import torch
import numpy as np


# First 

# df = torch.tensor([10, 20, 30, 40, 50])
# print(df)
# print(df.ndim)# this show how many dimenstion we have and on this part we have only one dimension so it should be 1 as i guess
# print(df.shape)
# print(df.dtype)



# print("\n\n",20*"=","Second Practice",20*"=")
# df = torch.tensor([
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9]
# ])
# print(df.ndim)
# print(df.shape)
# print(df.dtype)
# df = df.to(torch.float32)
# print(df.dtype)


# print("\n\n",20*"=","Third Practice",20*"=")
# x = torch.tensor([
#     [
#         [1, 2],
#         [3, 4]
#     ],
#     [
#         [5, 6],
#         [7, 8]
#     ]
# ])

# print(x.ndim)# => 3
# print(x.shape)#= > (2,2,2)
# print(x.dtype)# => int64


# Second

# #1
# df = torch.zeros(4,3)
# print(df)
# print(df.shape)
# print(df.ndim)
# print(df.dtype)


# #2
# df = torch.ones(5,2)
# df_norm = torch.randn(5,2)
# print(df)
# print(df.shape)
# print(df.ndim)
# print(df.dtype)
# print(df_norm)
# print(df_norm.shape)
# print(df_norm.ndim)
# print(df_norm.dtype)


# #3

# tens = torch.tensor([])
# for i,j in zip([2, 4, 6, 8, 10, 12, 14, 16, 18],range(9)):
#     tens[(j,)] = i
# print(tens)


#4
df = torch.linspace(0,1,5)
print(df)

#5
print(torch.arange(0, 1, 0.2))# چون که arrange میاد بینشون دوتا دوتتا دهم فاصله میندازه ولی اون یکی میاد بینش ۶ تا عدد پیدا میکنه
print(torch.linspace(0, 1, 6))

#6
df = torch.randn(4,3)
print(df)
print(df.shape)
print(df.ndim)
print(df.dtype)