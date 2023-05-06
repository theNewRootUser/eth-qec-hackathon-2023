# helper function to determine valid tic tac toe board positions
def get_winner(board):
        # Check the board for any winning combinations
    winning_combinations = [
        # Rows
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        # Columns
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        # Diagonals
        (0, 4, 8),
        (2, 4, 6),
    ]
    
    x_wins = False
    o_wins = False
    
    for combo in winning_combinations:
        if board[combo[0]] == board[combo[1]] == board[combo[2]] and board[combo[0]] != '':
            if board[combo[0]] == 'x':
                return [0,0,1]
            else:
                return [1,0,0]
    return [0,1,0]
    

def is_valid_tic_tac_toe(board):
    # Check that the board has exactly 9 elements
    if len(board) != 9:
        return False
    
    # Count the number of 'x' and 'o' on the board
    count_x = board.count('x')
    count_o = board.count('o')
    
    # Check that the difference in count between 'x' and 'o' is 0 or 1
    if abs(count_x - count_o) > 1:
        return False
    
    # Check the board for any winning combinations
    winning_combinations = [
        # Rows
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        # Columns
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        # Diagonals
        (0, 4, 8),
        (2, 4, 6),
    ]
    
    x_wins = False
    o_wins = False
    
    for combo in winning_combinations:
        if board[combo[0]] == board[combo[1]] == board[combo[2]] and board[combo[0]] != '':
            if board[combo[0]] == 'x':
                x_wins = True
            else:
                o_wins = True
    
    # Check if both 'x' and 'o' won or if neither won
    if x_wins and o_wins or (not x_wins and not o_wins):
        return False
    
    # Check that the board is a valid final board configuration
    if (x_wins and count_x != count_o + 1) or (o_wins and count_x != count_o):
        return False
    # All checks have passed, so the board is valid
    return True
  

def generate_tic_tac_toe_configs():
    valid_configs = []
    winners = []
    
    # Generate all possible configurations of the board
    for i in range(3**9):
        board = []
        for j in range(9):
            symbol = ''
            if i % 3 == 0:
                symbol = 'x'
            elif i % 3 == 1:
                symbol = 'o'
            board.append(symbol)
            i //= 3
        
        # Check if the configuration is valid
        if is_valid_tic_tac_toe(board):
            valid_configs.append(board)
            winners.append(get_winner(board))
    
    return valid_configs, winners



boards, winners = generate_tic_tac_toe_configs()
boards_t = []
for b in boards:
    data_t = np.array([1 if entry == 'x' else -1 if entry == 'o' else 0 for entry in b]) * 2 * np.pi / 3
    boards_t.append(data_t)
    
boards = np.array(boards_t)


boards = np.array(boards)
winners = np.array(winners)


import numpy as np
from qiskit import *

def encode_data_param(circuit):
    for i in range(9):
        circuit.rx(Parameter(f'data{i}'), i)


def add_single_qubit_gates(params, circuit):
    corner_qubits = [0,2,6,8]
    edge_qubits = [1,3,5,7]
    center_qubit = 4
    # corners (green)
    for index in corner_qubits:
        circuit.rx(params[0], index)
        circuit.ry(params[1], index)
    # edges (red)
    for index in edge_qubits:
        circuit.rx(params[2], index)
        circuit.ry(params[3], index)
    
    # middle (yellow)
    circuit.rx(params[4], center_qubit)
    circuit.ry(params[5], center_qubit)
    
    return circuit


def add_two_qubit_gates(params, circuit):
    # corners (green)
    corner_qubits = [0,2,6,8]
    edge_qubits = [1,3,5,7]
    center_qubit = 4
    
    # yellow two-qubit gates
    for index in corner_qubits:
        circuit.cp(params[0], center_qubit, index)
    # red two-qubit gates
    
    for index in edge_qubits:
        circuit.cp(params[1], index, center_qubit)
    
    # green two-qubit gates
    circuit.cp(params[2], 0, 1)
    circuit.cp(params[2], 2, 1)
    circuit.cp(params[2], 2, 5)
    circuit.cp(params[2], 8, 5)
    circuit.cp(params[2], 8, 7)
    circuit.cp(params[2], 6, 7)
    circuit.cp(params[2], 6, 3)
    circuit.cp(params[2], 0, 3)
    
    return circuit


x = boards
y = winners

# shuffle the indices
shuffle_indices = np.random.permutation(len(x))
train_size = int(len(x) * 0.3)

# split the indices into training and testing sets
train_indices = np.array(shuffle_indices[:train_size])
test_indices = np.array(shuffle_indices[train_size:])

# create the training and testing sets
x_train, y_train = np.take(x, train_indices, axis=0), np.take(y, train_indices, axis=0)
x_test, y_test = np.take(x, test_indices, axis=0), np.take(y, test_indices, axis=0)


from qiskit.circuit import Parameter

circ = QuantumCircuit(9)


encode_data_param(circ)
add_single_qubit_gates([Parameter(f'par{i}') for i in range(6)], circ)
add_two_qubit_gates([Parameter(f'par{i}') for i in range(6,9)], circ)


from qiskit import BasicAer, execute
from qiskit.algorithms.optimizers import SPSA
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator

observables = (
    SparsePauliOp("ZIZIIIZIZ") / 4.0,
    SparsePauliOp("IZIZIZIZI") / 4.0,
    SparsePauliOp("IIIIZIIII")
)

estimator = Estimator()

estimator_qnn = EstimatorQNN(
    circuit=circ,
    input_params=FEATURE_MAP.parameters,
    weight_params=VAR_FORM.parameters,
    estimator=estimator,
    observables=observables
)

from qiskit.algorithms.optimizers import L_BFGS_B
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier

initial_point = np.random.random(9)*2*np.pi

regressor = NeuralNetworkClassifier(
    neural_network=estimator_qnn,
    optimizer=SPSA(maxiter=5),
    initial_point=initial_point,
    one_hot=True,
)
regressor.fit(x_train[:10], y_train[:10])
regressor.score(x_test, y_test)