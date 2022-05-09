import numpy as np
from cmath import inf

from task_1 import get_inv

# Строим базисную матрицу
def get_basis_matrix(A, a_tran, B):
    A_b = np.zeros((A.shape[0], len(B)))
    i = 0
    for index in B:
        A_b[i] = a_tran[index]
        i += 1
    return A_b.transpose()

# Строим базисный вектор
def get_basis_vector(c, B):
    i = 0
    c_b = [0 for _ in B]
    for index in B:
            c_b[i] = c[index]
            i += 1
    return c_b

# Основная фаза симплекс метода
def simplex_method(c, A, x, B):
    count, A_b = 0, None
    a_tran = A.transpose()
    while True:
        # Шаги 1-2. Получаем матрицу, обратную к базисной, и вектор базисных компонент 
        if count == 0:
            A_b = get_basis_matrix(A, a_tran, B)
            A_b_inv = np.linalg.inv(A_b)
        else:
            A_b_inv = get_inv(A_b, A_b_inv, A[k])

        i = 0
        c_b = get_basis_vector(c, B)

        # Шаги 3-4. Находим векторы потенциалов и оценок
        u_t = np.dot(c_b, A_b_inv)
        delta = np.dot(u_t, A) - c;

        # Шаг 5. Проверка текущего плана на оптимальность
        neg_delta, j0 = None, None
        for i in range(len(delta)):
            # Шаг 6. Поиск отрицательной компоненты в векторе оценок
            if delta[i] < 0:
                neg_delta = delta[i]
                j0 = i
                break
        if neg_delta is None:
            return x, B

        # Шаги 7-8. Находим векторы z и Θ
        z = np.dot(A_b_inv, a_tran[j0])

        theta = np.zeros(len(z))
        for index, item in enumerate(z):
            if item <= 0:
                theta[index] = inf
            else:
                theta[index] = x[B[index]] / z[index]

        # Шаги 9-10. Нахождение минимума в Θ и проверка ограниченности целевой функции
        theta_0 = min(theta)
        if theta_0 == inf:
            print('Target function is not limited!')
            return None

        # Шаги 11-13. Преобразование базисного допустимого плана
        k = np.argmin(theta)
        j_star = B[k]
        B[k] = j0

        x[j0] = theta_0
        for i in range(A.shape[0]):
            if i != k:
                x[B[i]] -= theta_0 * z[i]
        x[j_star] = 0
