fully observable condition
measurements are power injections at each bus

input x: load demands at each bus; dimension [num_nodes, 2]
output y: voltage magnitude and voltage phasor angle at each bus; dimension [num_nodes, 2]
## output y: voltage magnitude and voltage phasor angle at each bus, except slack bus; pq and qg at slack bus; dimension [num_nodes, 2]

in mlp.py gnn.py: adding framework to customize neural network structure


