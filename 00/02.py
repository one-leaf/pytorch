import torch

def gradtest1():
    x = torch.ones(3,4,requires_grad=True)
    print("x",x)

    y = torch.ones(x.size()) *2 
    print("y",y)

    out = torch.mean(x * y) +1

    print("out",out)

    # out.backward() = out.backward(torch.tensor(1.))
    out.backward()

    # grad: d(out)/dx 
    # d(out)/d(x) ==> (1/(3*4) sum(2x) + 1 )/ d(x) ==>  1/12 *2 = 1/6

    print("y.grad", y.grad)
    print("x.grad", x.grad)

def gradtest2():
    x = torch.ones(3,4,requires_grad=True)
    print("x",x)

    y = torch.ones(x.size()) *2 
    print("y",y)

    out = torch.mean(x * y) +1

    out.backward(torch.tensor(3.0))
    # grad: d(out)/dx 
    # d(out)/d(x) ==> (1/(3*4) sum(2x) + 1 )/ d(x) *3 ==>  1/12 *2 *3= 1/2 = 0.5

    print("y.grad", y.grad)
    print("x.grad", x.grad)


def gradtest3():
    x = torch.randn(3, requires_grad=True)

    y = x * 2
    while y.data.norm() < 1000:
        y = y * 2

    print(y)

    v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
    y.backward(v)

    print(x.grad)

    print(x.requires_grad)
    print((x ** 2).requires_grad)

    with torch.no_grad():
        print((x ** 2).requires_grad)

if __name__ == "__main__":
    gradtest2()