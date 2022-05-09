import numpy as np

from task_2 import simplex_method, get_basis


c = np.array([1, 0, 0])
A = np.array([[1, 1, 1], 
             [2, 2, 2]])
b = np.array([0, 0])


def get_basis_plan(c, A, b):

    for index, elem in enumerate(b):
        if elem < 0:
            elem *= -1
            A[index] *= - 1 

    m = A.shape[0] 
    n = A.shape[1]

    c_dashed = np.array([0 if i < n else -1 for i in range(n + m)])
    a_dashed = np.append(A, np.eye(2), axis=1)
    x_dashed = np.append(np.array([0 for _ in range(n)]), b)
    B = np.array([n + i for i in range(m)])

    x, B = simplex_method(c_dashed, a_dashed, x_dashed, B)
    a_tran = a_dashed.transpose()
    a_basis_inv = np.linalg.inv(get_basis(a_dashed, a_tran, B))
    print(a_basis_inv)

    for i in range(m):
        if x[n + i] != 0:
            print("The problem is not joint!")
            exit(0)

    x = x[:n]
    while True:
        basis = [j <= n for j in B]
        if all(basis):
            print('B is good basis plan')
            return x, B

        k = np.argmax(B)
        j_k = B[k]

        l = {}
        for j in range(n): 
            if j not in B:
                l[j] =  np.dot(a_basis_inv, a_tran[j])

        for j in l:
            if l[j][k] == 0:
                A = np.delete(A, j, 0)
                a_dashed = np.delete(a_dashed, j, 0)
                B = np.delete(B, k)
                break


print(get_basis_plan(c, A, b))
