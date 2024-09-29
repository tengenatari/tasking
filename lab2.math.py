def alpha(matrix: list[list], vector: list):
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
    matrix, vector = alpha(matrix, vector)
    x = [vector[i] for i in range(len(vector))]
    new_x = [vector[i] for i in range(len(vector))]
    while True:
        for i in range(len(matrix)):
            new_x[i] = []



