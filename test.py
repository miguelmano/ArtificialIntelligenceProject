import filecmp
import time

import numpy as np

import utils
from search import (
    astar_search,
    depth_first_tree_search,
    greedy_search,
    InstrumentedProblem, breadth_first_tree_search
)
from takuzu import Takuzu, Board, TakuzuState

NUMBER_TESTS = 19
INPUT_FILE = "testes-takuzu/input_T{:02d}"
OUTPUT_FILE = "testes-takuzu/output_T{:02d}"
RESULT_FILE = "testes-takuzu/result_T{:02d}"


def parse_instance_from_file(path):
    with open(path, "r") as file:
        size = int(file.readline())
        board_list = []

        for i in range(size):
            line = file.readline()
            board_list.append([int(n) for n in line.split(sep='\t')])

    return Board(np.array(board_list).astype(int))


def print_to_file(path, board):
    with open(path, "w") as file:
        file.write(str(board))
        file.write("\n")


def execute_example(example: int):
    if example == 1:
        # Ler tabuleiro do ficheiro 'input_T00' (Figura 1):
        # $ python3 takuzu < input_T00
        board = parse_instance_from_file('testes-takuzu/input_T00')
        print("Initial:\n", board, sep="")
        # Imprimir valores adjacentes
        print(board.adjacent_vertical_numbers(3, 3))
        print(board.adjacent_horizontal_numbers(3, 3))
        print(board.adjacent_vertical_numbers(1, 1))
        print(board.adjacent_horizontal_numbers(1, 1))
    elif example == 2:
        # Ler tabuleiro do ficheiro 'i1.txt' (Figura 1):
        # $ python3 takuzu < i1.txt
        board = parse_instance_from_file('testes-takuzu/input_T00')
        print("Initial:\n", board, sep="")
        # Criar uma instância de Takuzu:
        problem = Takuzu(board)
        # Criar um estado com a configuração inicial:
        initial_state = TakuzuState(board)
        # Mostrar valor na posição (2, 2):
        print(initial_state.board.get_number(2, 2))
        # Realizar acção de inserir o número 1 na posição linha 2 e coluna 2
        result_state = problem.result(initial_state, (2, 2, 1))
        # Mostrar valor na posição (2, 2):
        print(result_state.board.get_number(2, 2))
    elif example == 3:
        # Ler tabuleiro do ficheiro 'i1.txt' (Figura 1):
        # $ python3 takuzu < i1.txt
        board = parse_instance_from_file('testes-takuzu/input_T00')
        # Criar uma instância de Takuzu:
        problem = Takuzu(board)
        # Criar um estado com a configuração inicial:
        s0 = TakuzuState(board)
        print("Initial:\n", s0.board, sep="")
        # Aplicar as ações que resolvem a instância
        s1 = problem.result(s0, (0, 0, 0))
        s2 = problem.result(s1, (0, 2, 1))
        s3 = problem.result(s2, (1, 0, 1))
        s4 = problem.result(s3, (1, 1, 0))
        s5 = problem.result(s4, (1, 3, 1))
        s6 = problem.result(s5, (2, 0, 0))
        s7 = problem.result(s6, (2, 2, 1))
        s8 = problem.result(s7, (2, 3, 1))
        s9 = problem.result(s8, (3, 2, 0))
        # Verificar se foi atingida a solução
        print("Is goal?", problem.goal_test(s9))
        print("Solution:\n", s9.board, sep="")
    elif example == 4:
        # Ler tabuleiro do ficheiro 'i1.txt' (Figura 1):
        # $ python3 takuzu < i1.txt
        board = parse_instance_from_file('testes-takuzu/input_T00')
        # Criar uma instância de Takuzu:
        problem = Takuzu(board)
        # Obter o nó solução usando a procura em profundidade:
        goal_node = depth_first_tree_search(problem)
        # Verificar se foi atingida a solução
        print("Is goal?", problem.goal_test(goal_node.state))
        print("Solution:\n", goal_node.state.board, sep="")


def execute_examples():
    for i in range(1, 5):
        execute_example(i)


def check_solutions():
    for i in range(NUMBER_TESTS):
        print("Starting {:02d}...".format(i), end="")

        board = parse_instance_from_file(INPUT_FILE.format(i))
        problem = Takuzu(board)
        goal_node = greedy_search(problem)
        solution = goal_node.state.board

        print_to_file(RESULT_FILE.format(i), solution)
        if filecmp.cmp(OUTPUT_FILE.format(i), RESULT_FILE.format(i)):
            print(" Finished with success")
        else:
            print(" Error")


def check_time():
    def compare_searchers(problems, header, searchers):
        table = []
        for searcher in searchers:
            row = [utils.name(searcher)]
            for problem in problems:
                start = time.time()
                searcher(problem)
                end = time.time()
                row.append(end - start)
            row.append(sum(row[1:]))
            table.append(row)
        utils.print_table(table, header, numfmt="{:.3f}")

    compare_searchers(
        problems=[Takuzu(parse_instance_from_file(INPUT_FILE.format(i))) for i in range(NUMBER_TESTS)],
        header=["Algorithm"] + ["T{:02d}".format(i) for i in range(NUMBER_TESTS)] + ["Total"],
        searchers=[
            breadth_first_tree_search,
            depth_first_tree_search,
            greedy_search,
            astar_search
        ]
    )


def check_space():
    def compare_searchers(problems, header, searchers):
        table = []
        for searcher in searchers:
            row = [utils.name(searcher)]
            for problem in problems:
                p = InstrumentedProblem(problem)
                searcher(p)
                row.append('<{}/{}>'.format(p.states, p.succs))
            table.append(row)
        utils.print_table(table, header)

    compare_searchers(
        problems=[Takuzu(parse_instance_from_file(INPUT_FILE.format(i))) for i in range(NUMBER_TESTS)],
        header=["Algorithm"] + ["T{:02d}".format(i) for i in range(NUMBER_TESTS)],
        searchers=[
            breadth_first_tree_search,
            depth_first_tree_search,
            greedy_search,
            astar_search
        ]
    )
    print("Caption: <generated/expanded>")


if __name__ == "__main__":
    # execute_examples()
    check_solutions()
    print("")
    check_time()
    print("")
    check_space()
