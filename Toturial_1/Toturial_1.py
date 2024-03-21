import torch

# Part1: Linear Math
print("==== Linear Math ====")
a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
b = torch.arange(12, dtype=torch.float32).new_ones(3, 4)
print("a = ", a)
print("b = ", b)
c = a + b
print("c = ",c)


x = torch.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [8, 7, 6, 5],
    [4, 3, 2, 1]])
y = torch.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [8, 7, 6, 5],
    [4, 3, 2, 1]])

print("x * y(0) = ", x*y)
print("x * y(1) = ", torch.mm(x, y))


u = torch.tensor([3.0, -4.0])
print(torch.norm(u))

 
# Part2: calculus Math
print("==== calculus Math ====")

def f(x):
    return 3*x**2 - 4*x
def numerical_lim(f, x, h):
    return (f(x+h) - f(x)) / h
h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    h *= 0.1
