from typing import Union

import torch
from torch import nn
import torch_geometric.nn as gnn

Activation = Union[str, nn.Module]
Gnn_layer = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}

_str_to_gnnlayer = {
    'gcn': gnn.GCNConv,
    'gat': gnn.GATConv,
    'gatv2': gnn.GATv2Conv,
}


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'leaky_relu',
        output_activation: Activation = 'identity',
) -> nn.Module:
    """
        Builds a feedforward neural network

        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            MLP (nn.Module)
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    # TODO: return a MLP. This should be an instance of nn.Module
    # Note: nn.Sequential is an instance of nn.Module.
    ## raise NotImplementedError
    modules = []
    modules.append(nn.Linear(input_size,size))
    modules.append(activation)
    modules.append(nn.Dropout( p = 0.5 ))
    for i in range(n_layers-2):
        modules.append(nn.Linear(size,size))
        modules.append(activation)
    modules.append(nn.Linear(size,output_size))
    modules.append(output_activation)
    model = nn.Sequential(*modules)
    return model

def build_gnn(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        input_args: str = 'x, edge_index',
        gnn_layer_name: str = 'gcn',
        activation: Activation = 'leaky_relu',
        output_activation: Activation = 'identity',
) -> nn.Module:
    """
        Builds a feedforward neural network

        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            MLP (nn.Module)
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    # TODO: return a MLP. This should be an instance of nn.Module
    # Note: nn.Sequential is an instance of nn.Module.
    ## raise NotImplementedError
    assert(gnn_layer_name in _str_to_gnnlayer)
    gnn_layer = _str_to_gnnlayer[gnn_layer_name]
    modules = []
    modules.append((gnn_layer(input_size,size), f'{input_args} -> x'))
    modules.append(activation)
    modules.append(nn.Dropout( p = 0.5 ))
    for i in range(n_layers-2):
        modules.append((gnn_layer(size,size), f'{input_args} -> x'))
        modules.append(activation)
    modules.append((gnn_layer(size,output_size), f'{input_args} -> x'))
    modules.append(output_activation)
    model = gnn.Sequential(input_args, modules)
    return model

device = None

def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
