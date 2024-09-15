import copy
import numpy


def pick_row(matrix: list[list], ind):
    for x in range(ind, len(matrix)):
        if matrix[x][ind] != 0:
            matrix[ind][ind], matrix[x][ind] = matrix[x][ind], matrix[ind][ind]
            return matrix, True
    return matrix, False


def matrix_input():
    n = int(input())
    matrix = []
    for x in range(n):
        matrix.append(list(map(float, input().split())))
    return matrix


def pick_abs(row: list[float]):
    row = row[:len(row) - 1]
    if abs(max(row)) > abs(min(row)):
        return row.index(max(row))
    return row.index(min(row))


def vector_input():
    vector = list(map(float, input().split()))
    return vector


def merge_mat_vec(matrix: list[list], vector: list):
    for x in range(len(vector)):
        matrix[x].append(vector[x])
    return matrix


def first_move(matrix: list[list]):
    for i in range(len(matrix)):
        matrix, flag = pick_row(matrix, i)
        if not flag:
            print('Many solutions')
            return matrix
        for j in range(i + 1, len(matrix)):
            el_ = matrix[j][i]
            for k in range(i, len(matrix[0])):
                matrix[j][k] -= el_ * matrix[i][k] / matrix[i][i]
    return matrix


def check_row(matrix: list[list], ind: int):
    if len(matrix[ind]) - matrix[ind].count(0) == 2:
        return True
    elif len(matrix[ind][:len(matrix[0]) - 1]) - matrix[ind].count(0) == 1:
        print('No solutions')
        return False
    elif matrix[ind].count(0) == len(matrix[ind]):
        print('Many solutions')
        return False
    return True


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


def second_move(matrix: list[list]):
    for i in range(len(matrix) - 1, -1, -1):
        for j in range(i + 1, len(matrix[0]) - 1):
            matrix[i][-1] -= matrix[j][-1]*matrix[i][j]
            matrix[i][j] = 0.
        if not check_row(matrix, i):
            return matrix
        matrix[i][-1] = matrix[i][-1]/matrix[i][i]
        matrix[i][i] = 1.
    return matrix


def gauss_solution(mat: list[list] = None, vec: list = None):
    if not mat:
        mat = matrix_input()
    if not vec:
        vec = vector_input()
    matrix = copy.deepcopy(mat)
    print(mat)
    mat = first_move(merge_mat_vec(mat, vec))
    mat1 = copy.deepcopy(mat)
    for row in mat:
        print(*row)

    mat = second_move(mat)

    for row in mat:
        print(*row)
    solution_vector = []
    for x in mat:
        solution_vector.append(x[-1])
    print('Solution vector:', solution_vector)
    return solution_vector, matrix, mat1


def gauss_solution_wme():

    mat = matrix_input()

    vec = vector_input()
    matrix = copy.deepcopy(mat)

    mat = wme_move(merge_mat_vec(mat, vec))
    for row in mat:
        print(*row)
    dic = dict()
    for x in range(len(mat)):
        for y in range(len(mat[0]) - 1):
            if mat[x][y] != 0:
                dic[y] = mat[x][-1]
    solution_vector = []
    print(dic)
    for x in range(len(dic)):
        solution_vector.append(dic[x])

    print('Solution_vector: ', solution_vector)
    return solution_vector, matrix, mat


def count_nevyazka(vector: list, matrix: list[list], *args):
    new_vector = copy.deepcopy(vector)
    for x in range(len(matrix)):
        for y in range(len(matrix[0])):
            new_vector[y] -= matrix[x][y] * vector[y]

    return new_vector


def count_e(vector: list, *args):
    return (sum(list(map(lambda x: x**2, vector))))**(1/2)





def count_det(mat: list[list] = None):
    if not mat:
        mat = matrix_input()
    _, _, mat = gauss_solution(mat.copy())
    det = 1
    for x in range(len(mat)):
        det *= mat[x][x]

    return det


def count_inv(mat: list[list] = None):
    inv_mat = [[0] * len(mat) for x in range(len(mat))]
    vector = [0] * len(mat)
    vector[-1] = 1
    for x in range(len(mat)):
        vector[x] = 1
        vector[x - 1] -= 1
        sol_vector, _, _ = gauss_solution(copy.deepcopy(mat), vector.copy())
        print(mat)
        for y in range(len(sol_vector)):
            inv_mat[y][x] = sol_vector[y]
    return inv_mat

def task1234():
    print(vector_nev := count_nevyazka(*gauss_solution_wme()))
    print(count_e(vector_nev))
    print(vector_nev := count_nevyazka(*gauss_solution()))
    print(count_e(vector_nev))

task1234()
count_inv(matrix_input())
