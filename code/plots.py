import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns  # Seaborn for better aesthetics
# DONE: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
	# DONE: Make plots for loss curves and accuracy curves.
	# DONE: You do not have to return the plots.
	# DONE: You can save plots as files by codes here or an interactive way according to your preference.
	
	epochs = range(1, len(train_losses) + 1)
    
    # Plotting loss curves
	plt.figure(figsize=(12, 5))
	plt.subplot(1, 2, 1)
	plt.plot(epochs, train_losses, label='Training Loss')
	plt.plot(epochs, valid_losses, label='Validation Loss')
	plt.title('Loss Curves')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()

	# Plotting accuracy curves
	plt.subplot(1, 2, 2)
	plt.plot(epochs, train_accuracies, label='Training Accuracy')
	plt.plot(epochs, valid_accuracies, label='Validation Accuracy')
	plt.title('Accuracy Curves')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.legend()

	plt.tight_layout()
	plt.show()


def plot_confusion_matrix(results, class_names, title='Normalized Confusion matrix', cmap=plt.cm.Blues):
	# TODO: Make a confusion matrix plot.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.

	# Unpack test results
    y_true, y_pred = zip(*results)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Normalize by row
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plotting
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap=cmap, xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
