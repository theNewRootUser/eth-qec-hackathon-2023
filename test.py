import numpy as np
from qiskit import *

from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator


corner_qubits = [0,2,6,8]
edge_qubits = [1,3,5,7]
center_qubit = 4

# tic_tac_toe_field = ['x','x','o','','o','x','o','','']
def encode_data(tic_tac_toe_field, circuit):
    data_g = [1 if entry == 'x' else -1 if entry == 'o' else 0 for entry in tic_tac_toe_field ]
    for entry, index in zip(data_g, range(len(data_g))):
        circuit.rx(entry * 2 * np.pi / 3, index)
                   
    return circuit



def add_single_qubit_gates(params, circuit):
    corner_params = params[0:2]
    edgeparams = params[2:4]
    middleparam = params[4:]
    print(corner_params)

    
    for i in corner_qubits:
        circuit.ry(corner_params[0], i)
        circuit.rx(corner_params[1], i)
    for i in edge_qubits:
        circuit.ry(edgeparams[0], i)
        circuit.rx(edgeparams[1], i)


    circuit.ry(middleparam[0], 4)
    circuit.rx(middleparam[1], 4)
    # edges (red)
    
    # middle (yellow)
    
    return circuit

def add_two_qubit_gates(params, circuit):
    # corners (green)
    
    
    # yellow two-qubit gates
        
    # red two-qubit gates
    
    # green two-qubit gates
    
    
    return circuit




# Create a Quantum Circuit acting on a quantum register of nine qubits
circ = QuantumCircuit(9)

encode_data(['x','x','o','','o','x','o','',''], circ)

circ.barrier()

single_qubit_gate_params=[0.2,0.3,0.2,0.4,0.5, 0.6]
add_single_qubit_gates(single_qubit_gate_params, circ)

circ.barrier()

two_qubit_gate_params=[0.2,0.3,0.2]
add_two_qubit_gates(two_qubit_gate_params, circ)

print(circ.draw())




estimator = Estimator()

circuits = (
    circ,
    circ,
    circ
)
observables = (
    SparsePauliOp("ZIZIZIZII") / 4.0,
    SparsePauliOp("IZIZIZIZI") / 4.0,
    SparsePauliOp("IIIIIIIIZ")
)

job = estimator.run(circuits, observables)
result = job.result()

print(f">>> Observables: {[obs.paulis for obs in observables]}")
print(f">>> Expectation values: {result.values.tolist()}")



