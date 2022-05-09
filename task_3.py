import numpy as np

from task_2 import simplex_method, get_basis


c = np.array([1, 0, 0])
A = np.array([[1, 1, 1], 
             [2, 2, 2]])
b = np.array([0, 0])


def get_basis_plan(c, A, b):
    # Шаг 1. Преобразование задачи для соблюдения неотрицательности
    for index, elem in enumerate(b):
        if elem < 0:
            elem *= -1
            A[index] *= - 1 

    m = A.shape[0] 
    n = A.shape[1]

    # Шаг 2. Составим вспомогательную задачу линейного программирования
    c_dashed = np.array([0 if i < n else -1 for i in range(n + m)])
    a_dashed = np.append(A, np.eye(2), axis=1)

    # Шаг 3. Построим начальный базисный допустимы план
    x_dashed = np.append(np.array([0 for _ in range(n)]), b)
    B = np.array([n + i for i in range(m)])

    # Шаг 4. Решим вспомогательную задачу основной фазой симплекс-метода
    x, B = simplex_method(c_dashed, a_dashed, x_dashed, B)
    a_tran = a_dashed.transpose()
    a_basis_inv = np.linalg.inv(get_basis(a_dashed, a_tran, B))
    print(a_basis_inv)

    # Шаг 5. Проверка условия совместности
    for i in range(m):
        if x[n + i] != 0:
            print("The problem is not joint!")
            return None

    # Шаг 6. Формируем допустимый план задачи
    x = x[:n]
    while True:
        # Шаг 7. Проверка допустимости текущего базисного планач    
        basis = [j <= n for j in B]
        print(basis)
        if all(basis):
            print('B is good basis plan')
            return x, B

        # Шаг 8. Находим максимальный индекс искусственной переменной
        k = np.argmax(B)

        # Шаг 9. Находим векторы l для каждого индекса от 1 до n, которого нет в В
        l = {}
        for j in range(n): 
            if j not in B:
                l[j] =  np.dot(a_basis_inv, a_tran[j])
        
        # Шаг 10. Преобразование множества базисных индексов
        for j in l:
            if l[j][k] != 0:
                B[k] = j

        # Шаг 11. Удаление линейно зависимых ограничений
        for j in l: 
            if l[j][k] == 0:
                A = np.delete(A, j, 0)
                a_dashed = np.delete(a_dashed, j, 0)
                b = np.delete(b, j)
                B = np.delete(B, j)
                break
