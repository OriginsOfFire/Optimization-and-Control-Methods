import numpy as np

from task_2 import get_basis_matrix, get_basis_vector


def dual_simplex_method(c, A, b, B):
    a_tran = A.transpose()
    A_b = None
    n = A.shape[1]
    while True:
        # Шаги 1-2. Находим матрицу, обратную базисной, и вектор базисных компонент
        A_b = get_basis_matrix(A, a_tran, B)
        A_b_inv = np.linalg.inv(A_b)
        c_b = get_basis_vector(c, B)

        # Шаг 3. Найдем базисный допустимый план двойственной задачи.
        y = np.dot(c_b, A_b_inv)

        # Шаг 4. Находим псевдоплан, соответствующий текущему базисному допустимому
        k_b = np.dot(A_b_inv, b)
        K = np.array([0 for _ in range(n)], dtype=np.float64)
        i = 0
        for index in B:
            K[index] = k_b[i]
            i += 1 

        # Шаг 5. Проверка псевдоплана на оптимальность 
        for elem in K:
            if elem < 0:
                break
        else:
            return K, B

        # Шаг 6. Находим отрицательную компоненту псевдоплана 
        j_k = np.argmin(K)
        k = None
        for index, elem in enumerate(B):
            if elem == j_k:
                k = index
        delta_y = A_b_inv[k]

        # Шаг 7. Вычисляем μ для каждого небазисного индекса
        mu = {}
        for j in range(n):
            if j not in B:
                mu[j] = np.dot(delta_y, a_tran[j])

        # Шаг 8. Проверка совместности прямой задачи
        check = [el >= 0 for el in mu.values()]
        if all(check):
            print('The direct problem is not joint!')
            return None, None

        # Шаг 9. Находим σ для каждого небазисного индекса, для которого μ отрицательно
        sigma = []
        for key in mu:
            if mu[key] < 0:
                tmp = (c[key] - np.dot(a_tran[key], y)) / mu[key]
                sigma.append(tmp)
        sigma = np.asarray(sigma)

        # Шаг 10. Находим индекс, на котором достигается минимум в σ
        j_0 = sigma.argmin()

        # Шаг 11. Заменяем k-й базисный индекс на j0 в B
        B[k] = j_0
