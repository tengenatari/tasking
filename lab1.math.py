
def matrix_input():
    n = int(input())
    matrix = []
    for x in range(n):
        matrix.append(list(map(float, input().split())))
    return matrix


def vector_input():
    vector = list(map(float, input().split()))
    return vector


def merge_mat_vec(matrix: list[list], vector: list):
    for x in range(len(vector)):
        matrix[x].append(vector[x])
    return matrix


def first_move(matrix: list[list]):
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            for k in range(i, len(matrix[0])):
                matrix[j][k] -= matrix[j][i] * matrix[i][k] / matrix[i][i]
    return matrix




def check_row(matrix: list[list], ind: int):
    if len(matrix[ind]) - matrix[ind].count(0) == 2:
        return True
    elif len(matrix[ind]) - matrix[ind].count(0) == 1:
        print('No solutions')
        return False
    print('Many solutions')
    return False


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


mat = matrix_input()

vec = vector_input()

mat = first_move(merge_mat_vec(mat, vec))

for row in mat:
    print(*row)


mat = second_move(mat)

for row in mat:
    print(*row)


