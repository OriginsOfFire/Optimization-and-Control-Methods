import numpy as np

from task_2 import get_basis_matrix, get_basis_vector


def minimize(A, c, x, D, j_b, j_b_ext):
    while True:
            A_b = A[:, j_b]
            A_b_inv = np.linalg.inv(A_b)

            # Нахождение промежуточных векторов и проверка оптимальности
            c_dash = c + np.dot(x, D)
            u_dash = np.dot(-c_dash[j_b], A_b_inv)
            delta_dash = np.dot(u_dash, A) + c_dash

            if (delta_dash >= 0).all():
                return x
            
            j0 = np.argmax(delta_dash < 0)

            # Находим  вектор l
            l = np.zeros(len(x))
            l[j0] = 1
            A_b_ext = A[:, j_b_ext]

            H = np.bmat([
                [D[j_b_ext, :][:, j_b_ext], A_b_ext.T],
                [A_b_ext, np.zeros((len(A), len(A)))]
            ])
            H_inv = np.linalg.inv(H)

            b_starred = np.concatenate((D[j_b_ext, j0], A[:, j0]))
            x_ = np.array(np.dot(-H_inv, b_starred))[0]
            l[:len(j_b_ext)] = x_[:len(j_b_ext)]
            
            # Находим Θ для каждого элемента множества расширенной опоры ограничений
            delta = np.dot(np.dot(l, D), l)
            theta = {}
            theta[j0] = np.inf if delta == 0 else np.abs(delta_dash[j0]) / delta
            
            for j in j_b_ext:
                if l[j] < 0:
                    theta[j] = -x[j] / l[j]
                else:
                    theta[j] = np.inf
            
            j_asterisk = min(theta, key=theta.get)
            theta_0 = theta[j_asterisk]

            # Проверка ограниченности целевой функции
            if theta_0 == np.inf:
                return None
            
            # Обновление допустимого плана
            x = x + theta_0 * l
            if j_asterisk == j0:
                j_b_ext.append(j_asterisk)
            elif j_asterisk in j_b_ext and j_asterisk not in j_b:
                j_b_ext.remove(j_asterisk)
            elif j_asterisk in j_b:
                third_condition = False
                s = j_b.index(j_asterisk)

                # Обновляем опоры ограничений
                for j_plus in set(j_b_ext).difference(j_b):
                    if (np.dot(A_b_inv, A[:, j_plus]))[s] != 0:
                        third_condition = True
                        j_b[s] = j_plus
                        j_b_ext.remove(j_asterisk)

                if not  third_condition:
                    j_b[s] = j0
                    j_b_ext[j_b_ext.index(j_asterisk)] = j0
