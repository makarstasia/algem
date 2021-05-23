import numpy as NX

import numpy as np
from numpy.dual import eigvals


def roots(p):
    """
    Возвращает корни многочлена с коэффициентами

    Значения в массиве `p` ранга 1 являются коэффициентами многочлена.
    Если длина `p` равна n + 1, то многочлен описывается следующим образом:

      p[0] * x**n + p[1] * x**(n-1) + ... + p[n-1]*x + p[n]

    :param
    ----------
    p : array
     Массив коэффициентов ранга 1.

    :return:
    -------
    out : ndarray
        Массив, содержащий корни многочлена.

    ------
    ValueError
        Когда `p` не может быть преобразован в массив ранга 1.

    дополнение
    --------
    poly : Найдите коэффициенты многочлена с заданной последовательностью корней.
    polyval : Вычислите значения полиномов.
    polyfit : Подбор полинома по методу наименьших квадратов.
    poly1d : Класс одномерных полиномов.



    """
    # Если входные данные скалярны, это делает их массивом
    p = np.atleast_1d(p)  # >>> np.atleast_1d(1, [3, 4])
    #   [array([1]), array([3, 4])]
    if p.ndim != 1:
        raise ValueError("Входные данные должны быть массивом с рангом 1..")

    # найти ненулевые значения массива
    non_zero = NX.nonzero(NX.ravel(p))[0]

    # Вернуть пустой массив, если в полиноме все нули
    if len(non_zero) == 0:
        return NX.array([])

    # найти количество конечных нулей - это количество корней в 0.
    trailing_zeros = len(p) - non_zero[-1] - 1

    # убрать начальные и конечные нули
    p = p[int(non_zero[0]):int(non_zero[-1]) + 1]

    # перевод в массив с переменными типа 'float'
    if not issubclass(p.dtype.type, (NX.floating, NX.complexfloating)):
        p = p.astype(float)

    N = len(p)
    if N > 1:
        # построить сопутствующую матрицу и найти ее собственные значения (корни)
        A = np.diag(NX.ones((N - 2,), p.dtype), -1)
        A[0, :] = -p[1:] / p[0]
        roots = eigvals(A)
    else:
        roots = NX.array([])

    # добавляем нули
    roots = np.hstack((roots, NX.zeros(trailing_zeros, roots.dtype)))
    return roots


def get_dimensions(matrix):
    """
    Возвращает размеры любой заданной матрицы
    :param matrix: Список списков `float` переменных.
    :return: список `float` значений, описывающих размеры матрицы
    """
    return [len(matrix), len(matrix[0])]


def find_determinant(matrix, excluded=1):
    """
    Возвращает значение определителя любой заданной матрицы
    :param matrix: Список списков `float` переменных.
    :param excluded: Значение `float`, которое относится к значению в
строке и столбце, вдоль которых были вычеркнуты элементы.
    :return: `float` значение, которое является определителем матрицы.
    """
    dimensions = get_dimensions(matrix)
    if dimensions == [2, 2]:
        return excluded * ((matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0]))
    else:
        new_matrices = []
        excluded = []
        exclude_row = 0
        for exclude_column in range(dimensions[1]):
            tmp = []
            excluded.append(matrix[exclude_row][exclude_column])
            for row in range(1, dimensions[0]):
                tmp_row = []
                for column in range(dimensions[1]):
                    if (row != exclude_row) and (column != exclude_column):
                        tmp_row.append(matrix[row][column])
                tmp.append(tmp_row)
            new_matrices.append(tmp)
        determinants = [find_determinant(new_matrices[j], excluded[j]) for j in range(len(new_matrices))]
        determinant = 0
        for i in range(len(determinants)):
            determinant += ((-1) ** i) * determinants[i]
        return determinant


def list_multiply(list1, list2):
    """
    Возвратите умножение двух списков, рассматривая каждый список как коэффициент.
    Например, чтобы умножить два списка длиной два, используйте метод FOIL.
    :param list1: list  c `float`
    :param list2: list  c `float`
    :return: Список значений с плавающей точкой, содержащий результат
               умножения.
    """
    result = [0 for _ in range(len(list1) + len(list2) - 1)]
    for i in range(len(list1)):
        for j in range(len(list2)):
            result[i + j] += list1[i] * list2[j]
    return result


def list_add(list1, list2, sub=1):
    """
    Вернуть поэлементное сложение двух списков
    :param list1: list  c `float`
    :param list2: list  c `float`
    :param sub: Значение int, на которое умножается каждый элемент во втором списке.
                По умолчанию установлено значение 1, а значение -1 приводит к вычитанию.
    :return: Список значений типа float, содержащий результат сложения.
    """
    return [i + (sub * j) for i, j in zip(list1, list2)]


def determinant_equation(matrix, excluded=[1, 0]):
    """
    Возвращает коэффициенты нек. уравнения
    Например, [1, 2, 3] соответствует уравнению
    1 + 2x + 3x^2.
    :param matrix: Список списков `float` переменных.
    :param excluded: Список значений `float`, который относится к значению в строке и столбце,
    вдоль которых были перечеркнуты элементы.
    :return: Список значений с `float` , соответствующих уравнению (как описано выше)
    """
    dimensions = get_dimensions(matrix)
    if dimensions == [2, 2]:
        tmp = list_add(list_multiply(matrix[0][0], matrix[1][1]), list_multiply(matrix[0][1], matrix[1][0]), sub=-1)
        return list_multiply(tmp, excluded)
    else:
        new_matrices = []
        excluded = []
        exclude_row = 0
        for exclude_column in range(dimensions[1]):
            tmp = []
            excluded.append(matrix[exclude_row][exclude_column])
            for row in range(1, dimensions[0]):
                tmp_row = []
                for column in range(dimensions[1]):
                    if (row != exclude_row) and (column != exclude_column):
                        tmp_row.append(matrix[row][column])
                tmp.append(tmp_row)
            new_matrices.append(tmp)
        determinant_equations = [determinant_equation(new_matrices[j],
                                                      excluded[j]) for j in range(len(new_matrices))]
        dt_equation = [sum(i) for i in zip(*determinant_equations)]
        return dt_equation


def identity_matrix(dimensions):
    """
    Возвращает единичную матрицу любых заданных размеров.
    :param dimensions: list  c `float`
    :return: список списков `float` со значениями 1 на главной диагонали.
    """
    matrix = [[0 for j in range(dimensions[1])] for i in range(dimensions[0])]
    for i in range(dimensions[0]):
        matrix[i][i] = 1
    return matrix


def characteristic_equation(matrix):
    """
    Возвращает характеристическое уравнение матрицы.
    :param matrix: список списков `float`.
    :return: Список списков значений типа float, содержащих характеристическое уравнение.
    """
    dimensions = get_dimensions(matrix)
    return [[[a, -b] for a, b in zip(i, j)] for i, j in zip(matrix,
                                                            identity_matrix(dimensions))]


def find_eigenvalues(matrix):
    """
    Возвращает собственные значения матрицы.
    :param matrix: список списков `float`.
    :return: Массив значений типа `float`, содержащий собственные значения.
    """
    dt_equation = determinant_equation(characteristic_equation(matrix))
    return np.roots(dt_equation[::-1])


A = [[6, 1, -1],
     [0, 7, 0],
     [3, -1, 2]]
eigenvalues = find_eigenvalues(A)
print(eigenvalues)
