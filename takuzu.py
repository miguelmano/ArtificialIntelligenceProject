# takuzu.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 49:
# 99286 Miguel Mano
# 99300 Pedro Rodrigues
import sys
from math import ceil

import numpy as np

from search import (
    Problem,
    Node,
    greedy_search
)


UNKNOWN = 0
UNSOLVED = 1
INVALID = 2
SOLVED = 3


class TakuzuState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = TakuzuState.state_id
        TakuzuState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    def __eq__(self, other):
        return self.board.__eq__(other.board)

    def __hash__(self):
        return self.board.__hash__()


class Board:
    """Representação interna de um tabuleiro de Takuzu."""

    def __init__(self, board: np.array):
        self._board = board
        self._state = UNKNOWN

        self.solve_trivial()

    def get_number(self, row: int, col: int) -> int:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self._board[row, col]

    def adjacent_vertical_numbers(self, row: int, col: int) -> (int, int):
        """Devolve os valores imediatamente abaixo e acima,
        respectivamente."""
        return (
            self.get_number(row + 1, col) if row < self.size() - 1 else None,
            self.get_number(row - 1, col) if row > 0 else None
        )

    def adjacent_horizontal_numbers(self, row: int, col: int) -> (int, int):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        return (
            self.get_number(row, col - 1) if col > 0 else None,
            self.get_number(row, col + 1) if col < self.size() - 1 else None
        )

    @staticmethod
    def parse_instance_from_stdin():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 takuzu.py < input_T01

            > from sys import stdin
            > stdin.readline()
        """
        size = int(sys.stdin.readline())
        board_list = [[int(n) for n in sys.stdin.readline().split(sep='\t')] for _ in range(size)]
        return Board(np.array(board_list).astype(int))

    def __eq__(self, other):
        return np.array_equal(self._board, other._board)

    def __hash__(self):
        return hash(self._board.data.tobytes())

    def __str__(self):
        return str(self._board).replace(' [', '').replace('[', '').replace(']', '').replace(' ', '\t')

    def size(self):
        return self._board.shape[0]

    def set_number(self, row, col, number):
        new_board = self._board.copy()
        new_board[row, col] = number
        return Board(new_board)

    def _empty_board(self):
        return self._board == 2

    def empty_indexes(self):
        return np.argwhere(self._empty_board())

    def number_empty(self):
        return self._empty_board().sum()

    def _transpose(self):
        self._board = self._board.transpose()

    def _count_per_row(self):
        return np.stack((
            (self._board == 0).sum(axis=1),
            (self._board == 1).sum(axis=1),
            (self._board == 2).sum(axis=1)
        ), axis=1)

    def _variable_heuristic_transposed(self):
        empty_board = self._empty_board()
        filled_board = ~empty_board

        b0 = filled_board[:-2].astype(int)
        b1 = filled_board[1:-1].astype(int)
        b2 = filled_board[2:].astype(int)
        constrained_triple = (b0 + b1 + b2) == 1

        number_constraints = np.zeros(self._board.shape)
        number_constraints[:-2] += constrained_triple
        number_constraints[1:-1] += constrained_triple
        number_constraints[2:] += constrained_triple

        filled_percentage = filled_board.sum(axis=1) / self.size()
        number_constraints += filled_percentage[:, None] / 2

        return number_constraints * empty_board

    def variable_heuristic(self):
        # Degree Heuristic
        number_constraints = self._variable_heuristic_transposed()
        self._transpose()
        number_constraints += self._variable_heuristic_transposed().transpose()
        self._transpose()

        return np.unravel_index(number_constraints.argmax(), number_constraints.shape)

    def _solve_trivial_transposed(self):
        board0 = self._board[:-2]
        board1 = self._board[1:-1]
        board2 = self._board[2:]

        def _propagate(b0, b1, b2):
            cond = (b0 != 2) & (b0 == b1) & (b2 == 2)
            b2 -= cond * (b0 + 1)

        # propagate 2 consecutive values
        _propagate(board0, board1, board2)
        _propagate(board1, board2, board0)

        # propagate to middle cell
        _propagate(board0, board2, board1)

        # finish columns with max amount of 0's or 1's
        cond0 = (self._board == 0).sum(axis=1) == ceil(self.size() / 2)
        cond1 = (self._board == 1).sum(axis=1) == ceil(self.size() / 2)
        values = cond0 + cond1 * 2
        self._board -= self._empty_board() * values[:, None]

    def solve_trivial(self):
        previous_board = None
        while not np.array_equal(previous_board, self._board):
            previous_board = self._board.copy()
            for _ in range(2):
                self._solve_trivial_transposed()
                self._transpose()

    def _solved(self):
        return not np.any(self._empty_board())

    def _invalid_transposed(self):
        # check solved rows uniqueness
        unique, counts = np.unique(self._board, axis=0, return_counts=True)
        if np.any((counts > 1) & ~np.any(unique == 2, axis=1)):
            return True

        # check number of 0's and 1's per row
        counts0 = (unique == 0).sum(axis=1)
        counts1 = (unique == 1).sum(axis=1)
        if np.any(np.concatenate((counts0, counts1)) > ceil(self.size() / 2)):
            return True

        # check for 3 consecutive elements in columns
        board0 = self._board[:-2]
        board1 = self._board[1:-1]
        board2 = self._board[2:]
        if np.any((board0 == board1) & (board1 == board2) & (board0 != 2)):
            return True

    def _invalid(self):
        if self._invalid_transposed():
            return True

        self._transpose()
        is_invalid = self._invalid_transposed()
        self._transpose()
        return is_invalid

    def _get_state(self):
        if self._state != UNKNOWN:
            return self._state

        if self._invalid():
            self._state = INVALID
        elif self._solved():
            self._state = SOLVED
        else:
            self._state = UNSOLVED

        return self._state

    def solved(self):
        return self._get_state() == SOLVED

    def invalid(self):
        return self._get_state() == INVALID


class Takuzu(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        super().__init__(TakuzuState(board))

    def actions(self, state: TakuzuState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        if state.board.invalid():
            return []

        index = state.board.variable_heuristic()
        return [(index[0], index[1], number) for number in [0, 1]]

    def result(self, state: TakuzuState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        new_board = state.board.set_number(action[0], action[1], action[2])
        return TakuzuState(new_board)

    def goal_test(self, state: TakuzuState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes."""
        return state.board.solved()

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        return node.state.board.number_empty()


if __name__ == "__main__":
    # Ler o ficheiro de input de sys.argv[1],
    board = Board.parse_instance_from_stdin()

    # Usar uma técnica de procura para resolver a instância,
    problem = Takuzu(board)
    goal_node = greedy_search(problem)

    # Retirar a solução a partir do nó resultante,
    solution = goal_node.state.board

    # Imprimir para o standard output no formato indicado.
    print(solution)
