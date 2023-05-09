import torch
from torch.utils.data import TensorDataset, DataLoader
from feature_engineering import DataProcessor
from LSTM import LSTMModel
from RNN import RNN
#from train import ModelTrainer
from training_prep import create_data_sequences
import matplotlib.pyplot as plt


# Hyperparameters
input_size = 100
hidden_size = 256
num_layers = 1
batch_size = 64
num_epochs = 10
learning_rate = 0.001

# Load and preprocess data
dataset_dir = 'data/'
data_processor = DataProcessor(dataset_dir)

# Choose to preprocess data or load saved mappings
preprocess_data = False
char_to_id_file = 'char_to_id.json'
id_to_char_file = 'id_to_char.json'

if preprocess_data:
    ids, char_to_id, id_to_char = data_processor.preprocess()
    data_processor.save_mappings(char_to_id_file, char_to_id, id_to_char_file, id_to_char)
else:
    char_to_id, id_to_char = data_processor.load_mappings(char_to_id_file, id_to_char_file)
    
## Create Dataset sequences with pytorch
dataloader = create_data_sequences(' '.join((data_processor.read_dialogue_lines()).values())[:1000], char_to_id, batch_size)

## Train and Evaluate the model
model = RNN(input_size, hidden_size, 100, num_layers)
losses = model.train(dataloader,model)

# Plot losses
plt.plot(losses)
plt.title('Training loss evolution')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
plt.savefig('losses.png')



