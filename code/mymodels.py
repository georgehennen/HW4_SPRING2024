import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.nn.utils import rnn as rnn_utils

##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####


class MyMLP(nn.Module):
	def __init__(self):
		super(MyMLP, self).__init__()
		self.layer1 = nn.Linear(178, 16)
		self.layer2 = nn.Linear(16, 5)
		self.sigmoid = nn.ReLU()

	def forward(self, x):
		x = self.sigmoid(self.layer1(x))
		x = self.layer2(x)

		return x


class MyCNN(nn.Module):
	def __init__(self):
		super(MyCNN, self).__init__()
		self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=8, stride=5)
		self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=8, stride=5)
		# Initialized to None, will be set on first forward pass
		self.fc1 = None  
		self.fc2 = None

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		if self.fc1 is None:
			# Calculate the size only once, assuming x's size does not change
			calculated_size = x.view(x.size(0), -1).size(1)
			print(calculated_size)
			# Now that we have calculated_size, create the fully connected layers
			self.fc1 = nn.Linear(in_features=calculated_size, out_features=128)
			self.fc2 = nn.Linear(in_features=128, out_features=5)
			# Copy the weights to the appropriate device (e.g. GPU)
			self.fc1 = self.fc1.to(x.device)
			self.fc2 = self.fc2.to(x.device)
		# Flatten the tensor for the fully connected layers
		x = x.view(-1, self.fc1.in_features)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)  # No ReLU after the final layer
		return x



class MyRNN(nn.Module):
    def __init__(self):
        super(MyRNN, self).__init__()
        self.num_layers = 2  # Number of stacked GRU layers
        self.hidden_size = 16  # Number of features in the hidden state
        # Define GRU layer
        self.gru = nn.GRU(input_size=1, hidden_size=16, num_layers=2, batch_first=True, dropout=0.5)
        # Define a fully connected layer
        self.fc = nn.Linear(16, 5)  # Mapping to 5 output classes

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Forward propagate through the GRU
        out, _ = self.gru(x, h0)
        # Take the output from the last time step
        out = self.fc(out[:, -1, :])
        return out


class MyVariableRNN(nn.Module):
	def __init__(self, dim_input):
		super(MyVariableRNN, self).__init__()
        # Define the first fully conne
		self.fc1 = nn.Linear(dim_input, 32)
        # Define the GRU layer
		self.gru = nn.GRU(32, 16, batch_first=True)
		self.dropout = nn.Dropout(p=0.2)
		self.fc2 = nn.Linear(16, 2)

	def forward(self, input_tuple):
		seqs, lengths = input_tuple

		x = torch.tanh(self.fc1(seqs))
		packed_x = rnn_utils.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

		packed_gru_out, _ = self.gru(packed_x)
		gru_out, _ = rnn_utils.pad_packed_sequence(packed_gru_out, batch_first=True)

		idx = (lengths - 1).view(-1, 1).expand(len(lengths), gru_out.size(2)).unsqueeze(1)
		idx = idx.to(gru_out.device)
		last_gru_out = gru_out.gather(1, idx).squeeze(1)
		last_gru_out = self.dropout(last_gru_out)
		out = self.fc2(last_gru_out)

		return out