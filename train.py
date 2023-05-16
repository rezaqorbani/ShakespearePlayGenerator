import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from utils.feature_engineering import DataProcessor
from models.models import LSTMModel
from trainers.trainer import ModelTrainer
import torch.nn as nn
from test import Evaluater


# Hyperparameters
input_size = 100
hidden_size = 256
num_layers = 1
batch_size = 64
num_epochs = 10
learning_rate = 0.001

# Load and preprocess data
dataset_dir = './data/Cornell_Movie-Dialog_Corpus/'
data_processor = DataProcessor(dataset_dir)

# Choose to preprocess data or load saved mappings
preprocess_data = False
char_to_id_file = './data/mappings/char_to_id.json'
id_to_char_file = './data/mappings/id_to_char.json'

if preprocess_data:
    ids, char_to_id, id_to_char = data_processor.preprocess()
    data_processor.save_mappings(char_to_id_file, char_to_id, id_to_char_file, id_to_char)
else:
    char_to_id, id_to_char = data_processor.load_mappings(char_to_id_file, id_to_char_file)
print('Mapping loaded')
## Create Dataset sequences with pytorch
#data_processor = DataProcessor(dataset_dir)
dialogue_lines = data_processor.read_dialogue_lines()
text = ' '.join(dialogue_lines.values())
ids = data_processor.text_to_ids(text, char_to_id)
print(len(ids))
dataset = data_processor.create_dataset(ids[:1000050])


train_loader, val_loader, test_loader=DataProcessor.create_loaders(dataset, 0.8, 0.1, 0.1, 50)
print('Data Loaded')
## Train and Evaluate the model
# Initialize the LSTM model
input_size = embedding_size = 50
hidden_size = 256
num_layers = 2
vocab_size = len(char_to_id)
model = LSTMModel(input_size, hidden_size, vocab_size, num_layers)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

# Define the loss function, learning rate, and optimizer
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
trainer=ModelTrainer(model, train_loader, criterion, optimizer, device)

print('start training')
# Training loop

num_epochs = 1
for epoch in range(num_epochs):
    loss = trainer.train()
    # Evaluate the model on the validation set
    validation_loss = trainer.evaluate(val_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Validation Loss: {validation_loss:.4f}")

# Evaluation on test set
test_loss = trainer.evaluate(test_loader)

print('Loss on testing data:', test_loss)

model=trainer.model

evaluater = Evaluater(model, device)

perplexity= evaluater.calculate_perplexity(test_loader, criterion)
print('Perplexity:' , perplexity)

seed_text = "Start a discussion about coffee"
gen_length = 2000

generated_text = evaluater.generate_text(seed_text, gen_length, char_to_id, id_to_char, device,  temperature=0.8, top_p=0.9)
print(generated_text)