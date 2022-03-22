import torch

a=torch.meshgrid(torch.arange(3), torch.arange(4))
a=torch.stack(a[::-1],dim=0).float()
b=a.repeat(1,1,1,1)
c=b[:,:1]
print(c.reshape(12,1,1,1))


