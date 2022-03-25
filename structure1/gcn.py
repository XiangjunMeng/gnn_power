import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import scipy.io as sci
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
import pytorch_util as pu

#thepath = '/home/hugo/experiment/gnn_power/acc/result/'
thepath = '/home/hugo/experiment/gnn_power/structure1/result/'
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
        x = pu.from_numpy(data_np[0]['x'][0,0])
        #print(x[0].shape)
        y = pu.from_numpy(data_np[0]['y'][0,0])
        edge_index = torch.from_numpy(data_np[0]['edge_index'][0,0]).long().to(pu.device)
        edge_attr = pu.from_numpy(data_np[0]['edge_attr'][0,0])
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
# loader = DataLoader(dataset, batch_size=32)
print()
# print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
#print(f'Number of features: {loader.num_features}')
#print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]
x_row, x_col = data.x.shape
y_row, y_col = data.y.shape

print()
#print(data)
print('===========================================================================================================')

# Gather some statistics about the graph.
print(f'data.x.shape: {x_row, x_col}')
print(f'data.y.shape: {y_row, y_col}')
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
from torch_geometric.nn import GCNConv

seed = 1234
batch_size = 32

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(seed)
        self.conv1 = GCNConv(x_col, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, y_col)
        self.lrelu = LeakyReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        # x = x.relu()
        x = self.lrelu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GCN(hidden_channels=32).to(pu.device)
print(model)

#criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
criterion = torch.nn.MSELoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.


all_data_num = len(dataset)
endi = int(np.ceil(all_data_num * 0.8))
train_dataset = dataset[0 : endi]
test_dataset = dataset[endi : all_data_num]
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# print(f'DEBUG {len(train_loader), len(train_loader.dataset)}')
print(f'DEBUG {len(test_loader), len(test_loader.dataset)}')

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         # out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         out = model(data.x, data.edge_index)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(loader):
    model.eval()

    mse_avg_sum = 0
    count = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        # out = model(data.x, data.edge_index, data.batch)  
        out = model(data.x, data.edge_index)
        #pred = out.argmax(dim=1)  # Use the class with highest probability.
        #correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        mse_avg_sum += criterion(out, data.y) 
        count += 1
    # print(f'DEBUG {out.shape, data.y.shape}')
    # print(f'DEBUG {debug}')
    return mse_avg_sum / count


for epoch in range(1, 171):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
