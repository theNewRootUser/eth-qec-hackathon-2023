import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
import tensorflow as tf
import matplotlib.pyplot as plt

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
                return [0, 0, 1]
            else:
                return [1, 0, 0]
    return [0, 1, 0]

#This function checks for a couple of things, length of the board,
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
    for i in range(3 ** 9):
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

import pennylane as qml
from pennylane import numpy as np

def encode_data(tic_tac_toe_field):
    # data_g = [1 if entry == 'x' else -1 if entry == 'o' else 0 for entry in tic_tac_toe_field]
    for entry, index in zip(tic_tac_toe_field, range(len(tic_tac_toe_field))):
        qml.RX(entry, wires=[index])
        #print(qml.RX(entry * 2 * np.pi / 3, wires=[index]))

    return


def add_single_qubit_gates(params):
    # define edges centers and lats.
    edges = [0, 2, 6, 8]
    lats = [1, 3, 5, 7]
    center = 4

    for i in edges:
        qml.RX(params[0], wires=[i])
        qml.RY(params[1], wires=[i])
    for i in lats:
        qml.RX(params[2], wires=[i])
        qml.RY(params[3], wires=[i])


    qml.RX(params[4], wires=[center])
    qml.RY(params[5], wires=[center])

    return

def add_two_qubit_gates(params):
    # corners (green)
    corner_qubits = [0, 2, 6, 8]
    edge_qubits = [1, 3, 5, 7]
    center_qubit = 4


    # yellow two-qubit gates
    for i in range(4):
        qml.CPhase(params[2], wires=[4, edge_qubits[i]])


    # red two-qubit gates, hard coded
    qml.CPhase(params[1], wires=[1, 0])
    qml.CPhase(params[1], wires=[1, 2])
    qml.CPhase(params[1], wires=[3, 0])
    qml.CPhase(params[1], wires=[3, 6])
    qml.CPhase(params[1], wires=[5, 2])
    qml.CPhase(params[1], wires=[5, 8])
    qml.CPhase(params[1], wires=[7, 6])
    qml.CPhase(params[1], wires=[7, 8])

    # green two-qubit gates, hard coded
    qml.CPhase(params[0], wires=[0, 4])
    qml.CPhase(params[0], wires=[2, 4])
    qml.CPhase(params[0], wires=[6, 4])
    qml.CPhase(params[0], wires=[8, 4])

    return

### SO FAR SO GOOD ###

print("ok")

obs_ZIZIIIZIZ = 0.25 * qml.PauliZ(0) @ qml.PauliZ(2) @ qml.PauliZ(6) @ qml.PauliZ(8)
obs_IIIIZIIII = qml.PauliZ(4)
obs_IZIZIZIZI = 0.25 * qml.PauliZ(1) @ qml.PauliZ(3) @ qml.PauliZ(5) @ qml.PauliZ(7)

observables = (obs_ZIZIIIZIZ,)# obs_IIIIZIIII, obs_IZIZIZIZI)

dev = qml.device("default.qubit", wires=9)
@qml.qnode(dev)
def circuit(params, tactoe):
    encode_data(tactoe)
    add_single_qubit_gates(params[:6])
    add_two_qubit_gates(params[6:])
    return qml.expval(obs_ZIZIIIZIZ)#, qml.expval(obs_IIIIZIIII), qml.expval(obs_IZIZIZIZI)

tictac = ['x', 'o', '', '', '', '', '', '', '']
total_params=[0.2,0.3,0.2,0.4,0.5, 0.6, 0.2,0.3,0.2]

##SETUP OF THE LABELS AND TRAININGS

x = np.array([[0 if e == '' else 2*np.pi/3 if e == 'x' else -2*np.pi/3 for e in b] for b in boards])
y = np.array(winners)[:,0]

# shuffle the indices
shuffle_indices = np.random.permutation(len(x))
train_size = int(len(x) * 0.3)

# split the indices into training and testing sets
train_indices = np.array(shuffle_indices[:train_size])
test_indices = np.array(shuffle_indices[train_size:])

# create the training and testing sets
X, Y = np.take(x, train_indices, axis=0), np.take(y, train_indices, axis=0)
x_test, y_test = np.take(x, test_indices, axis=0), np.take(y, test_indices, axis=0)

##DEFINE A COST FUNCTION

def variational_classifier(weights, bias, x):
    return circuit(weights, x) + bias

def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss

def accuracy(labels, predictions):

    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)

    return loss

#check if the cost definition is correct
def cost(weights, bias, X, Y):
    predictions = [variational_classifier(weights, bias, x) for x in X]
    # print(predictions)
    return square_loss(Y, predictions)

##Actual Training of it
#Nesterov goes from 0.01 to 0.99, smaller values indicate smaller step size so lower computation speed but higher accuracy.
##Vice versa for bigger  values. Let's take a simple 0.5 for now. Also try PSPA and Adam.


bias_init = np.array(0.0, requires_grad=True)
opt = NesterovMomentumOptimizer(0.1)
batch_size = 10

from tqdm import tqdm

costo = []

params = total_params  #at some point we can start it randomly though now i can't
bias = bias_init
weights = np.random.rand(9)*2*np.pi
for it in tqdm(range(100)):
    acc=0
    batch_index = np.random.randint(0, len(X), (batch_size))
    X_batch = X[batch_index]
    Y_batch = Y[batch_index]
    weights, bias, _, _ = opt.step(cost, weights, bias, X_batch, Y_batch)
    print(weights)
    costo.append(cost(weights, bias, X, Y))
    pred_index = np.random.randint(0, len(X), (10))
    predictions = [np.sign(variational_classifier(weights, bias, x)) for x in X[pred_index]]
    acc = square_loss(Y, predictions)


    print(
        "Iter: {:5d} | Cost: {:0.7f} | Accuracy: {:0.7f} ".format(
            it + 1, cost(weights, bias, X, Y), acc
        )
    )

print(params)
plt.plot(x,y)
plt
