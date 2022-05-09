import numpy as np
from cmath import inf

from task_1 import get_inv


c = np.array([1, 1, 0, 0, 0])
A = np.array([
    [-1, 1, 1, 0, 0],
    [1, 0, 0, 1, 0],
    [0, 1, 0, 0, 1]
])

x = np.array([0, 0, 1, 3, 2])
B = np.array([2, 3, 4])

# Строим базисную матрицу
def get_basis(A, a_tran, B):
    A_b = np.zeros((A.shape[0], len(B)))
    i = 0
    for index in B:
        A_b[i] = a_tran[index]
        i += 1
    return A_b.transpose()

# Основная фаза симплекс метода
def simplex_method(c, A, x, B):
    #TODO: add matrix inversion method from lab 1
    while True:
        a_tran = A.transpose()
        A_b = get_basis(A, a_tran, B)

        # Получаем матрицу, обратную к базисной, и вектор базисных компонент 
        A_b_inv = np.linalg.inv(A_b)

        i = 0
        c_b = np.asarray([0 for _ in B])    
        for index in B:
            c_b[i] = c[index]
            i += 1


        # Находим векторы потенциалов и оценок
        u_t = np.dot(c_b, A_b_inv)
        delta = np.dot(u_t, A) - c;

        # Проверка текущего плана на оптимальность
        neg_delta, j0 = None, None
        for i in range(len(delta)):
            if delta[i] < 0:
                neg_delta = delta[i]
                j0 = i
                break
        if neg_delta is None:
            return x, B

        # Находим векторы z и Θ
        z = np.dot(A_b_inv, a_tran[j0])

        theta = np.zeros(len(z))
        for index, item in enumerate(z):
            if item <= 0:
                theta[index] = inf
            else:
                theta[index] = x[B[index]] / z[index]

        # Проверка ограниченности целевой функции
        theta_0 = min(theta)
        if theta_0 == inf:
            print('Target function is not limited!')
            return None

        # Преобразование базисного допустимого плана
        k = np.argmin(theta)
        j_star = B[k]
        B[k] = j0

        x[j0] = theta_0
        for i in range(A.shape[0]):
            if i != k:
                x[B[i]] -= theta_0 * z[i]
        x[j_star] = 0
    