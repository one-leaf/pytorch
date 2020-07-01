import torch

print(torch.empty(2,3))

print(torch.rand(2,3))

print(torch.tensor([[1,2],[3,4],[5,6]]))

x = torch.randn(2,3)
print(x)
x=x.new_ones(2,5)
print(x)
y=torch.rand_like(x)
print(y,y.size())

print(y.size()[0])

x = torch.rand(5, 3)
y = torch.rand(5, 3)

print("x+y",x+y)
print(torch.add(x,y))
x.add(y)
print("x.add(y)",x)
x.add_(y)
print("x.add_(y)",x)
x=x+y
print("x=x+y",x)
print("x[:,-1]",x[:,-1])

z = x.view(3,5)
print(z)

lr = torch.randn(1)
print(lr)
print(lr.item())