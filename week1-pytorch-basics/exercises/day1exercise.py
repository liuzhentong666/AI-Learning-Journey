import torch
eye1 =torch.eye(5)
print(eye1)
random_tensor=torch.rand(3,4)*2-1
print(random_tensor)
arrange_ans=torch.arange(0,101,5)
print((arrange_ans))
print("--"*50)
x = torch.arange(24).reshape(2, 3, 4)

# 1. 提取第一个"页"的所有数据
first_page = x[0]
print("第一个页的数据:\n", first_page)

# 2. reshape成(6, 4)
reshaped_x = x.reshape(6, 4)
print("\nreshape后的结果:\n", reshaped_x)

# 3. 计算每列的平均值
col_means = reshaped_x.float().mean(dim=0)
print("\n每列平均值:\n", col_means)
def f():
    randomtensor=torch.rand(3,4)
    def stand(tensor):
        mean=torch.mean(tensor)
        std=torch.std(tensor)
        return (tensor-mean)/std
    normalized=stand(randomtensor)
    return normalized
