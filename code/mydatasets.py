import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import TensorDataset, Dataset

##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####

def load_seizure_dataset(path, model_type):
	"""
	:param path: a path to the seizure data CSV file
	:return dataset: a TensorDataset consists of a data Tensor and a target Tensor
	"""

	# DONE: Read a csv file from path.
	# DONE: Please refer to the header of the file to locate X and y.
	# DONE: y in the raw data is ranging from 1 to 5. Change it to be from 0 to 4.
	# DONE: Remove the header of CSV file of course.
	# DONE: Do Not change the order of rows.
	# DONE: You can use Pandas if you want to.

	df=pd.read_csv(path)
	X = df.iloc[:, :-1].values
	y = df.iloc[:, -1].values

	y = y - 1

	X_tensor = torch.tensor(X, dtype=torch.float32)
	y_tensor = torch.tensor(y, dtype=torch.long)

	if model_type == 'MLP':
		pass
	elif model_type == 'CNN':
		X_tensor = X_tensor.unsqueeze(1) 
	elif model_type == 'RNN':
		X_tensor = X_tensor.view(X_tensor.size(0), 178, 1)
	else:
		raise AssertionError("Wrong Model Type!")

	dataset = TensorDataset(X_tensor, y_tensor)

	return dataset


def calculate_num_features(seqs):
	"""
	:param seqs:
	:return: the calculated number of features
	"""
	# TODO: Calculate the number of features (diagnoses codes in the train set)
	flat_list = [item for sublist in seqs for visit in sublist for item in visit]
	num_features = max(flat_list) + 1  # Assuming feature IDs start at 0
	return num_features


class VisitSequenceWithLabelDataset(Dataset):
	def __init__(self, seqs, labels, num_features):
		"""
		Args:
			seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
			labels (list): list of labels (int)
			num_features (int): number of total features available
		"""

		if len(seqs) != len(labels):
			raise ValueError("Seqs and Labels have different lengths")

		self.labels = labels

		# DONE: Complete this constructor to make self.seqs as a List of which each element represent visits of a patient
		# DONE: by Numpy matrix where i-th row represents i-th visit and j-th column represent the feature ID j.
		# DONE: You can use Sparse matrix type for memory efficiency if you want.
		self.seqs = []
		for seq in seqs:
			# Initialize a zero matrix for each patient's visit sequence
			patient_matrix = np.zeros((len(seq), num_features), dtype=np.float32)

			for i, visit in enumerate(seq):
				# Mark the features/codes present in this visit
				patient_matrix[i, visit] = 1.0

			# Convert the numpy matrix to a PyTorch tensor and append to the list
			self.seqs.append(torch.tensor(patient_matrix))

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		# returns will be wrapped as List of Tensor(s) by DataLoader
		return self.seqs[index], self.labels[index]


def visit_collate_fn(batch):
	"""
	DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
	Thus, 'batch' is a list [(seq_1, label_1), (seq_2, label_2), ... , (seq_N, label_N)]
	where N is minibatch size, seq_i is a Numpy (or Scipy Sparse) array, and label is an int value

	:returns
		seqs (FloatTensor) - 3D of batch_size X max_length X num_features
		lengths (LongTensor) - 1D of batch_size
		labels (LongTensor) - 1D of batch_size
	"""

	# TODO: Return the following two things
	# TODO: 1. a tuple of (Tensor contains the sequence data , Tensor contains the length of each sequence),
	# TODO: 2. Tensor contains the label of each sequence

	batch.sort(key=lambda x: len(x[0]), reverse=True)

	# Extract sequences and labels from the batch, and calculate lengths
	sequences, labels = zip(*batch)
	lengths = [len(seq) for seq in sequences]

	# Determine the maximum length of a sequence in this batch
	max_length = max(lengths)

	# Get the number of features from the first sequence
	num_features = sequences[0].shape[1]

	# Initialize padded sequences tensor with zeros
	seqs_tensor = torch.zeros(len(batch), max_length, num_features, dtype=torch.float)

	for i, (seq, length) in enumerate(zip(sequences, lengths)):
		# Note: Assuming seq is a numpy array. If it's a scipy sparse matrix, convert it to dense
		if hasattr(seq, "toarray"):  # Check if scipy sparse matrix and convert to dense
			seq = seq.toarray()
		seqs_tensor[i, :length] = torch.tensor(seq, dtype=torch.float)

	lengths_tensor = torch.tensor(lengths, dtype=torch.long)
	labels_tensor = torch.tensor(labels, dtype=torch.long)

	return (seqs_tensor, lengths_tensor), labels_tensor
