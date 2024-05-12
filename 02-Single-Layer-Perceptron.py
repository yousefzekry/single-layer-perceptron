import numpy as np

# Defining a function pre_out that gives the output of the activation


def pre_out(v):
    if v >= 0:
        return 1
    else:
        return 0


# Defining a function update that calculate v and updates the weights.

def update(x, w, d, b):

    dot = np.dot(w, x)+b
    v = pre_out(dot)
    error = d - v
    if (d > v):
        return w+(error*x)
    elif (d < v):
        return w+(error*x)
    elif (d == v):
        return w


lis = [[0, 0], [0, 1], [1, 0], [1, 1]]
test_x = np.array(lis)
d = [0, 0, 0, 1]
w = np.array([0, 2])
b = -3


for i in range(0, 8):
    n_i = i
    if (i > 3):
        n_i = i-(4*(i//4))
    print("iteration= ", i, " before update w=", w)
    print("Before update predicted value for x=",
          test_x[n_i], " y=", pre_out(np.dot(test_x[n_i], w)+b))
    w = update(test_x[n_i], w, d[n_i], b)
    print("iteration= ", i, " after update w=", w)
    print("After update predicted value for x=",
          test_x[n_i], " y=", pre_out(np.dot(test_x[n_i], w)+b))
    print("-------------------------------------------------")

    print("Final weight vector=", w)
print("Predicted output")
print("x1  x2    y")
for i in range(0, 4):
    y = pre_out(np.dot(test_x[i], w)+b)
    print(test_x[i][0], "  ", test_x[i][1], "  ", y)
