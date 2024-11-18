import copy

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

def pick_row(matrix: list[list], ind):
    for x in range(ind, len(matrix)):
        if matrix[x][ind] != 0:
            matrix[ind][ind], matrix[x][ind] = matrix[x][ind], matrix[ind][ind]
            return matrix, True
    return matrix, False


def first_move(matrix: list[list]):
    for i in range(len(matrix)):
        matrix, flag = pick_row(matrix, i)
        if not flag:
            return matrix
        for j in range(i + 1, len(matrix)):
            el_ = matrix[j][i]
            for k in range(i, len(matrix[0])):
                matrix[j][k] -= el_ * matrix[i][k] / matrix[i][i]
    return matrix


def second_move(matrix: list[list]):
    for i in range(len(matrix) - 1, -1, -1):
        for j in range(i + 1, len(matrix[0]) - 1):
            matrix[i][-1] -= matrix[j][-1] * matrix[i][j]
            matrix[i][j] = 0.
        if not check_row(matrix, i):
            return matrix
        matrix[i][-1] = matrix[i][-1] / matrix[i][i]
        matrix[i][i] = 1.
    return matrix


def gauss_solution(mat: list[list] = None):

    matrix = copy.deepcopy(mat)
    print(mat)

    mat1 = copy.deepcopy(mat)

    mat = second_move(mat)

    solution_vector = []
    for x in mat:
        solution_vector.append(x[-1])

    return solution_vector, matrix, mat1