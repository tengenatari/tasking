import copy
import numpy as np

def razn(a, b):
    new_vec = [0]*len(a)
    for x in range(len(a)):
        new_vec[x] = a[x] - b[x]
    return new_vec

def LU(A):
    L = [[0] * len(A) for _ in range(len(A))]
    U = [[0] * len(A) for _ in range(len(A))]
    for x in range(len(A)):
        L[x][x] = 1
    for i in range(len(A)):
        for j in range(len(A)):
            if i <= j:
                U[j][i] = A[i][j] - sum(list(map(lambda x, y: x * y, L[i][:i], U[j][:i])))
            else:
                L[i][j] = (A[i][j] - sum(list(map(lambda x, y: x * y, L[i][:j], U[j][:j]))))/ U[j][j]

    return L, U

def move(matrix):
    for i in range(len(matrix) - 1, -1, -1):
        for j in range(i + 1, len(matrix[0]) - 1):
            matrix[i][-1] -= matrix[j][-1] * matrix[i][j]
            matrix[i][j] = 0.
        if not check_row(matrix, i):
            return matrix
        matrix[i][-1] = matrix[i][-1] / matrix[i][i]
        matrix[i][i] = 1.
    solution_vector = []
    for row in matrix:
        print(*row)
    for x in matrix:
        solution_vector.append(x[-1])
    return solution_vector


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
def transpose(n_matrix: list[list]):
    matrix = copy.deepcopy(n_matrix)
    for i in range(len(matrix)):
        for j in range(i, len(matrix[0])):
            matrix[j][i], matrix[i][j] = matrix[i][j], matrix[j][i]
    return matrix

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

    print('Прямой ход')
    for row in mat:
        print(*row)
    print('Обратный ход')
    mat = second_move(mat)

    for row in mat:
        print(*row)
    solution_vector = []
    for x in mat:
        solution_vector.append(x[-1])
    print('Solution vector:', solution_vector)
    return solution_vector, matrix, mat1, vec


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

    for x in range(len(dic)):
        solution_vector.append(dic[x])
    print()
    print('Solution_vector: ', solution_vector)
    return solution_vector, matrix, mat, vec


def count_nevyazka(vector: list, matrix: list[list], mat, b):
    new_vector = copy.deepcopy(b)
    for x in range(len(matrix)):
        for y in range(len(matrix[0])):
            new_vector[x] -= matrix[x][y] * vector[y]

    return new_vector


def count_e(vector: list, *args):
    return (sum(list(map(lambda x: x**2, vector))))**(1/2)





def count_det(mat: list[list] = None):
    if not mat:
        mat = matrix_input()
    _, _, mat, _ = gauss_solution(mat.copy())
    det = 1
    for x in range(len(mat)):
        det *= mat[x][x]

    return det


def count_inv(mat: list[list] = None):
    inv_mat = [[0] * len(mat) for _ in range(len(mat))]
    vector = [0] * len(mat)
    vector[-1] = 1

    for k in range(len(mat)):
        vector[k] = 1
        vector[k - 1] -= 1
        sol_vector, _, _, _ = copy.deepcopy(gauss_solution(copy.deepcopy(mat), vector))

        for y in range(len(sol_vector)):
            inv_mat[y][k] = sol_vector[y]

    return inv_mat

def task1234():
    print('op element')
    print(vector_nev := count_nevyazka(*gauss_solution_wme()))
    print()
    print('Morma Nevyazka')

    print(count_e(vector_nev))
    print("Gauss Method")
    print(vector_nev := count_nevyazka(*gauss_solution()))
    print('Morma Nevyazka')
    print(count_e(vector_nev))
    print()

def matrixmult (A, B):
    C = [[0 for row in range(len(A))] for col in range(len(B[0]))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k]*B[k][j]
    return C
def count_e2(matrix):
    return sum([sum(list(map(lambda x: x ** 2, y))) for y in matrix]) ** (1 / 2)

"""

vec = vector_input()
dv = [0.1 * x**2 + 0.01*x + 0.0001 + vec[x] for x in range(len(vec))]
print(dv)
print("with op element")
smth = gauss_solution(copy.deepcopy(matrixxx), vec)
smth1 = gauss_solution(copy.deepcopy(matrixxx), dv)
print(vector_nev := count_nevyazka(*smth))
print(vector_nev2 := count_nevyazka(*smth))
print('dx = ', razn(smth[0], smth1[0]))
print('x = ', smth[0])
print("dx/x= ", count_e(razn(smth[0], smth1[0]))/count_e(smth[0]))
print("V(a) db/b = ", count_e(razn(dv, vec))/count_e(vec))
print('Morma Nevyazka')
print(count_e(vector_nev))
print(count_e(vector_nev2))
"""

matrixxx = matrix_input()
L, U = LU(matrixxx)
print('матрица L')
for x in range(len(matrixxx)):
    print(*L[x])
U = transpose(U)
print()
print('матрица U')
for x in range(len(matrixxx)):
    print(*U[x])
print()
print('LU = A')
print(matrixmult(L, U))

vec = vector_input()
print()
L = L[::-1]

for x in range(len(matrixxx)):
    L[x] = L[x][::-1]

new_vec = vec[::-1]
new_vec = move(merge_mat_vec(L, new_vec))
print('Ly = b')
print(new_vec)
print()
new_vec = new_vec[::-1]
sol_vec = move(merge_mat_vec(U, new_vec))
print('Ux = y')
print(sol_vec)

print('невязка')
print(vector_nev := count_nevyazka(sol_vec, matrixxx, 0, vec))
