Graph Neural Networks for Power System State Estimation
This repository explores graph neural networks (GNNs) for learning power-system behavior from network topology and bus-level measurements. It includes Graph Convolutional Networks (GCNs), a Graph Attention Network (GAT), and multilayer perceptron (MLP) baselines.
Overview:
Power grids are naturally represented as graphs, with buses as nodes and transmission or distribution lines as edges. The models learn mappings from bus-level active and reactive load demands to power-system outputs using simulated data generated with MATLAB and MATPOWER.

The repository contains three experimental configurations:
- structure1/ — Predicts bus-level power generation from load-demand inputs.
- structure2/ — Predicts bus voltage magnitudes and phase angles from load-demand inputs.
- structure3/ — Extends voltage-state prediction experiments with configurable GNN architectures and a GAT model using edge attributes.

Methods and Tools
- Models: GCN, GAT, and MLP
- Frameworks: PyTorch, PyTorch Geometric
- Data generation: MATLAB, MATPOWER
- Graph features: Bus-level active/reactive loads, grid connectivity, and branch attributes such as resistance and reactance
- Training: Supervised regression using mean squared error (MSE)

Repository Structure
acc/          MATLAB scripts for generating power-system scenarios
structure1/   Generation prediction experiments
structure2/   Voltage magnitude and angle prediction experiments
structure3/   Configurable GNN and attention-based experiments
Each experimental directory contains MATLAB data-generation scripts and Python implementations of neural-network models.

Getting Started
1. Install Python and the required libraries: torch, torch-geometric, numpy, scipy, matplotlib, and scikit-learn.
2. Install MATLAB and MATPOWER to generate simulation datasets.
3. Update the MATPOWER paths in the MATLAB scripts and generate the corresponding .mat dataset files.
4. Place the generated dataset in the relevant experiment's result/ directory, then run a model from that directory, for example:
cd structure3
python gcn.py
# or
python gatconv.py
Note: Some scripts use hard-coded dataset names and local paths; these may need adjustment for your environment. The repository does not provide a turnkey installation or a verified end-to-end reproduction script.
Applications
Graph-based learning for power-flow approximation, electrical-grid state prediction, and data-driven power-system analysis.
