import copy
import numpy as np


def mu(matrix):
    max_ = -1000000000000
    for i in range(len(matrix)):
        if (1 - sum(list(map(abs, matrix[i][:i])))) != 0:
            max_ = max(max_, sum(list(map(abs, matrix[i][i:])))/(1 - sum(list(map(abs, matrix[i][:i])))))
    return max_


def alpha(matrix: list[list], vector: list):
    matrix = copy.deepcopy(matrix)
    vector = copy.deepcopy(vector)

    for i in range(len(matrix)):
        vector[i] /= matrix[i][i]
        matrix[i] = list(map(lambda x: -x/matrix[i][i], matrix[i]))
        matrix[i][i] = 0
    return matrix, vector


def first_and_inf_norma(matrix: list[list]):
    max_1 = sum(list(map(abs, matrix[0])))
    max_2 = sum([abs(matrix[x][0]) for x in range(len(matrix))])
    for x in range(len(matrix) - 1):
        max_1 = max(max_1, sum(list(map(abs, matrix[x]))))
        max_2 = max(max_2, sum([abs(matrix[y][x]) for y in range(len(matrix))]))
    return max_1, max_2


def subtract_vectors(x, y):
    new_vector = [x[i] - y[i] for i in range(len(x))]
    return new_vector


def solve_easy_iteration(matrix: list[list], vector: list, eps):

    x = [vector[i] for i in range(len(vector))]
    new_x = [vector[i] for i in range(len(vector))]
    cnt = 0
    while True:
        cnt += 1
        for i in range(len(matrix)):
            new_x[i] = sum(list(map(lambda z, y: z*y, matrix[i], x))) + vector[i]

        if min(first_and_inf_norma([subtract_vectors(new_x, x)])) < eps:
            print("Количество итераций", cnt)
            return new_x
        x = copy.deepcopy(new_x)


def solve_zeidel(matrix: list[list], vector: list, eps):

    x = [vector[i] for i in range(len(vector))]
    new_x = [vector[i] for i in range(len(vector))]
    cnt = 0
    while True:
        cnt += 1
        for i in range(len(matrix)):
            new_x[i] = (sum(list(map(lambda z, y: z * y, matrix[i][:i], new_x[:i]))) +
                        sum(list(map(lambda z, y: z * y, matrix[i][i:], x[i:]))) +
                        vector[i])

        if min(first_and_inf_norma([subtract_vectors(new_x, x)])) < eps:
            print("Количество итераций", cnt)
            return new_x
        x = copy.deepcopy(new_x)


def count_nevyazka(vector: list, matrix: list[list], b):
    new_vector = copy.deepcopy(b)
    for x in range(len(matrix)):
        for y in range(len(matrix[0])):
            new_vector[x] -= matrix[x][y] * vector[y]

    return new_vector


def merge_mat_vec(matrix: list[list], vector: list):
    for x in range(len(vector)):
        matrix[x].append(vector[x])
    return matrix


def pick_abs(row: list[float]):
    row = row[:len(row) - 1]
    if abs(max(row)) > abs(min(row)):
        return row.index(max(row))
    return row.index(min(row))


def wme_move(matrix: list[list]):

    for i in range(len(matrix)):
        ind = pick_abs(matrix[i])
        if matrix[i][ind] == 0:
            print('Many solutions')
            return matrix
        matrix[i] = list(map(lambda x: x / matrix[i][ind], matrix[i]))
        for j in range(len(matrix)):
            if j != i:
                el_ = matrix[j][ind]
                for k in range(len(matrix[0])):
                    matrix[j][k] -= el_ * matrix[i][k] / matrix[i][ind]
    return matrix


def gauss_solution_wme(matrix, vector):

    mat = copy.deepcopy(matrix)

    vec = copy.deepcopy(vector)

    mat = wme_move(merge_mat_vec(mat, vec))

    for row in mat:
        print(*row)
    dic = dict()
    for x in range(len(mat)):
        for y in range(len(mat[0]) - 1):
            if mat[x][y] != 0:
                dic[y] = mat[x][-1]
    solution_vector = []

    for x in range(len(dic)):
        solution_vector.append(dic[x])
    print()
    return solution_vector


def main(easy=True):
    print('Вводим размерность матрицы')
    n = int(input())
    print()

    print('Матрица')
    matrix = [list(map(float, input().split())) for x in range(n)]
    print()

    print('Вводим вектор')
    vector = list(map(float, input().split()))
    print()

    print('Вид после приведения матрицы к решаемому виду')
    matrix_s, vector_s = alpha(matrix, vector)
    print('матрица')
    print(np.array(matrix_s))
    print('вектор')
    print(np.array(vector_s))
    print()

    print('Считаем норму для матрицы')

    print(eps := first_and_inf_norma(matrix_s))
    print()

    def cont_sol(solution):
        print('Считаем неувязочку')

        print(count_nevyazka(solution, matrix, vector))

        print()
        print('Решаем методом гаусса с выбором главного элемента')

        print(abs_solution := gauss_solution_wme(matrix, vector))
        print()
        print('Считаем неувязочку для Гаусса')

        print(count_nevyazka(abs_solution, matrix, vector))
        print()
        print('dx')
        print(sub_vec := subtract_vectors(abs_solution, solution))
        print()
        print('Абсолютная погрешность')
        print(first_and_inf_norma([sub_vec])[0])
        print()
        print('Относительная погрешность')
        print(first_and_inf_norma([sub_vec])[0] / first_and_inf_norma([abs_solution])[0])
        print()
        print('')

    print('Решаем систему уравнений')

    print('Метод простых итераций')
    print('q = ', eps)
    print(solution := solve_easy_iteration(matrix_s, vector_s, (1 - min(eps))/min(eps) * 10**(-4)))
    cont_sol(solution)
    print('eps:', 10**(-4))
    print()
    print('Метод Зейделя')
    eps = mu(matrix_s)
    print('mu', eps)
    print(solution := solve_zeidel(matrix_s, vector_s, (1 - eps)/eps * 10**(-4)))
    print('eps', 10**(-4))
    cont_sol(solution)
    print()


main(easy=False)



