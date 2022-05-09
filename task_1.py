import numpy as np


def get_inv(A, A_inv, x):
    i = len(x) - 1
    # Замена i-го столбца на вектор-столбец x
    for j in range(len(x)):
        A[j][i] = x[j]
    # Шаг 1. Находим l = A_inv * x и проверяем, существует ли обратная матрица.
    l = np.matmul(A_inv, x)
    if l[i] == 0:
        return None

    # Шаг 2. В копии вектора l заменим i-й элемент на -1
    l_copy = np.copy(l)
    l_copy[i] = -1

    # Шаг 3. Находим l_dashed
    l_dashed = (-1 / l[i]) * l_copy

    # Шаг 4. Матрица Q формируется из единичной матрицы порядка n заменой i-го столбца на l_dashed
    Q = np.eye(3)
    for j in range(len(l_dashed)):
        Q[j][i] = l_dashed[j]

    #Шаг 5. Находим обратную матрицу путем умножения Q на A_inv 
    res = np.zeros([3,3])
    for j in range(len(x)):
        for k in range(len(x)):
            first = Q[j][j] * A_inv[j][k]
            second = Q[j][i] * A_inv[i][k]
            res[j][k] = first + second if j != k else first
    return res


A = np.array([[1, -1, 0], [0, 1, 0], [0, 0, 1]])
A_inv = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
x = np.array([1, 0, 1])