import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from utils.feature_engineering import MovieDialogsLoader
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
data_processor = MovieDialogsLoader(dataset_dir, level='word')  # set level to 'word'

# Load Word2Vec model
word2vec_model = Word2Vec.load('path_to_your_word2vec_model')  # Replace with your model path
embedding_size = word2vec_model.vector_size

# Choose to preprocess data or load saved mappings
preprocess_data = False
word_to_id_file = './data/mappings/word_to_id.json'
id_to_word_file = './data/mappings/id_to_word.json'

if preprocess_data:
    ids, word_to_id, id_to_word = data_processor.preprocess()
    data_processor.save_mappings(word_to_id_file, word_to_id, id_to_word_file, id_to_word)
else:
    word_to_id, id_to_word = data_processor.load_mappings(word_to_id_file, id_to_word_file)

# Initialize the weight for unknown words
unknown_vector = torch.empty(1, embedding_size).uniform_(-0.1, 0.1)

# Create an embedding matrix
vocab_size = len(word_to_id)
embedding_matrix = torch.cat([unknown_vector, torch.zeros(1, embedding_size)])  # For <PAD> and <UNK> tokens

for word in word_to_id:
    if word in word2vec_model:
        vector = torch.FloatTensor(word2vec_model[word])
        embedding_matrix = torch.cat([embedding_matrix, vector.unsqueeze(0)], 0)
    else:
        embedding_matrix = torch.cat([embedding_matrix, unknown_vector], 0)

# Create Dataset sequences with pytorch
dialogue_lines = data_processor.read_dialogue_lines()
text = ' '.join(dialogue_lines.values())
ids = data_processor.text_to_ids(text, word_to_id)
dataset = data_processor.create_dataset(ids[:1000050])

train_loader, val_loader, test_loader = MovieDialogsLoader.create_loaders(dataset, 0.8, 0.1, 0.1, 50)

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