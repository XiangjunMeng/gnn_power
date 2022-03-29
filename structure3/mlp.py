import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import scipy.io as sci
import numpy as np
from torch_geometric.data import Data
import pytorch_util as pu

#thepath = '/home/hugo/experiment/gnn_power/acc/result/'
#thepath = '/home/hugo/experiment/gnn_power/structure2/result/'
thepath = './result/'
casefile = 'case12da'
matdata_file = thepath + casefile + '_data.mat'
pu.init_gpu()

def load_matdata():
    mat = sci.loadmat(matdata_file)
    dataset_np = mat['dataset']# directly loaded as numpy array
    #print(type(dataset_np))
    num_data, _ = dataset_np.shape
   
    all_data = []
    print(num_data)

    for loop0 in range(0, num_data):
        data_np = dataset_np[loop0]
        #print(len(data), data[0])
        #print(data[0]['x'])
        x = data_np[0]['x'][0,0]
        #print(x[0].shape)
        y = data_np[0]['y'][0,0]
        edge_index = data_np[0]['edge_index'][0,0]
        edge_attr = data_np[0]['edge_attr'][0,0]
        cur_data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
        cur_data.num_nodes = x.shape[0]
        #print(x.shape[1])
        cur_data.node_feature = x.shape[1] # Note that node_feature is not Data's default parameters (added by myself)
        all_data.append(cur_data)
    
    #print(type(x[0,0]), x[0,0].shape)
    return all_data

def convert_data(dataset):
    data = dataset[0]
    x_row, x_col = data.x.shape
    y_row, y_col = data.y.shape
    batch_size = len(dataset)
    nn_x = np.zeros((batch_size, x_row * x_col))
    nn_y = np.zeros((batch_size, y_row * y_col))
    for loop0 in range(0, batch_size ):
        data = dataset[loop0]
        nn_x[loop0, :] = np.reshape(data.x, x_row * x_col, order = 'F')
        nn_y[loop0, :] = np.reshape(data.y, y_row * y_col, order = 'F')

    return nn_x, nn_y
    

dataset = load_matdata()
print()
# print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
#print(f'Number of features: {dataset.num_features}')
#print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]
y_row, y_col = data.y.shape

print()
#print(data)
print('===========================================================================================================')

# Gather some statistics about the graph.
#print(f'Number of nodes: {data.num_nodes}')
#print(f'Number of edges: {data.num_edges}')
#print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
#print(f'Number of training nodes: {data.train_mask.sum()}')
#print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
#print(f'Has isolated nodes: {data.has_isolated_nodes()}')
#print(f'Has self-loops: {data.has_self_loops()}')
#print(f'Is undirected: {data.is_undirected()}')


import torch
from torch.nn import Linear
from torch.nn import LeakyReLU
import torch.nn.functional as F

seed = 1234

class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(seed)
        self.lin1 = Linear(data.num_nodes * data.node_feature, hidden_channels) # in the original data, I still put x,y as [num_nodes, *], here I need to reshape them into a single-dimension vector
        self.lin2 = Linear(hidden_channels, y_row * y_col)
        self.lrelu = LeakyReLU()
    def forward(self, x):
        # print('calling MLP forward')
        x = self.lin1(x)
        # x = x.relu()
        x = self.lrelu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x


# model = MLP(hidden_channels=16).to(pu.device)
model = pu.build_mlp(input_size = data.num_nodes * data.node_feature, \
                    output_size = y_row * y_col, \
                    n_layers = 2, \
                    size = 16, \
                    activation = 'leaky_relu', \
                    output_activation = 'identity').to(pu.device)
print(model)
#criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
criterion = torch.nn.MSELoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.

nn_x, nn_y = convert_data(dataset)

#print(nn_x[0,:], dataset[0].x)
all_data_num, _ = nn_x.shape
endi = int(np.ceil(all_data_num * 0.8))
nn_x_train = pu.from_numpy(nn_x[0 : endi , :])
nn_y_train = pu.from_numpy(nn_y[0 : endi , :])
nn_x_test = pu.from_numpy(nn_x[endi : all_data_num , :])
nn_y_test = pu.from_numpy(nn_y[endi : all_data_num , :])

# print(nn_x_train.is_cuda, nn_y_train.is_cuda, nn_x_test.is_cuda, nn_y_test.is_cuda)
#print(nn_x_train.device, nn_y_train.device, nn_x_test.device, nn_y_test.device, model.dummy_param.device)
print(nn_x_train.device, nn_y_train.device, nn_x_test.device, nn_y_test.device)

def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(nn_x_train)  # Perform a single forward pass.
    # print('out.is_cuda ', out.is_cuda)
    loss = criterion(out, nn_y_train)  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss

def test(label_data_x, label_data_y):
    model.eval()
    # out = model(nn_x_test)
    out = model(label_data_x)
    # test_mse = (np.square(out - nn_y_test)).mean(axis=None) 
    # test_mse = criterion(out, nn_y_test) 
    test_mse = criterion(out, label_data_y) 
    return test_mse

n_poch = 201
for epoch in range(1, n_poch):
    loss = train()
    train_acc = test(nn_x_train, nn_y_train)
    test_acc = test(nn_x_test, nn_y_test)

    # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    

#test_acc = test()
#print(f'Test Accuracy: {test_acc:.4f}')
