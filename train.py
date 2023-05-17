import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from data_loader.data_loaders import CornellMovieDialogsLoader
from models.models import LSTMModel
from gensim.models import Word2Vec
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
data_loader = CornellMovieDialogsLoader(dataset_dir)  # set level to 'word'
word2vec_model = data_loader.word2vec
embedding_size = data_loader.embedding_dim

text_word_ids, word_to_id, id_to_word, embedding_matrix, vocab_size = data_loader.preprocess()
dataset = data_loader.create_dataset(text_word_ids[:1000])
train_loader, val_loader, test_loader = data_loader.create_loaders(dataset, 0.8, 0.1, 0.1, 50)

print("data loaded")

# Initialize the LSTM model with an embedding layer
model = LSTMModel(input_size, hidden_size, vocab_size, num_layers, embedding_size, embedding_matrix)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the loss function, learning rate, and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=word_to_id['<PAD>'])  # Ignore padding tokens
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
trainer = ModelTrainer(model, train_loader, criterion, optimizer, device)

# Training loop
for epoch in range(num_epochs):
    print("starting epoch", epoch+1)
    loss = trainer.train()
    validation_loss = trainer.evaluate(val_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Validation Loss: {validation_loss:.4f}")

## Evaluate the model

test_loss = trainer.evaluate(test_loader)

evaluater = Evaluater(model, device)

perplexity = evaluater.calculate_perplexity(test_loader, criterion)
print('Perplexity:', perplexity)

seed_text = "Start a discussion about coffee"
gen_length = 2000

generated_text = evaluater.generate_text(seed_text, gen_length, word_to_id, id_to_word, device, temperature=0.8, top_p=0.9)
print(generated_text)