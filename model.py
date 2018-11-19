import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    
    def __init__(self, state_size, action_size, seed, hidden_layers, drop_p):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers (list): size of hidden_layers
            drop_p (float): probability of an element to be zeroed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.dropout = nn.Dropout(p=drop_p)
        self.output = nn.Linear(hidden_layers[-1], action_size)

    def forward(self, state):
        
        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        for linear in self.hidden_layers:
            state = F.relu(linear(state))
            state = self.dropout(state)
            
        return self.output(state)
    
    
class DuelingNet(nn.Module):
    
    def __init__(self, state_size, action_size, seed, hidden_layers, drop_p):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers (list): size of hidden_layers of the common part
            drop_p (float): probability of an element to be zeroed
        """
        super(DuelingNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.dropout = nn.Dropout(p=drop_p)
        
        self.fc1_adv = nn.Linear(hidden_layers[-1], 64)
        self.fc2_adv = nn.Linear(64, action_size)
        
        self.fc1_val = nn.Linear(hidden_layers[-1], 64)
        self.fc2_val = nn.Linear(64, 1)

    def forward(self, state):

        for linear in self.hidden_layers:
            state = F.relu(linear(state))
            state = self.dropout(state)
            
        A = F.relu(self.fc1_adv(state))
        A = self.dropout(A)
        A = F.relu(self.fc2_adv(A))
        
        V = F.relu(self.fc1_val(state))
        V = self.dropout(V)
        V = F.relu(self.fc2_val(V))
            
        return V.expand_as(A) + A - A.mean(1).unsqueeze(1).expand_as(A)
    
    
    
