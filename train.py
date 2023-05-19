import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from data_loader.data_loaders import ShakespearePlaysLoader
from models.models import LSTMModel, RNNModel
from trainers.trainer import ModelTrainer
import torch.nn as nn
from tqdm import tqdm
#from test import Evaluater
from utils.augmenter import TextAugmenter
from test import Evaluater




def train(dataset_dir, level='char', model_name='RNN', batch_size=64, train_split=0.8, val_split=0.1, num_epochs=2, learning_rate=0.001, hidden_size=256, embedding_size=100, num_layers=1, input_size=100, dataset_length=1000, use_bpe=False, use_augmentation=False):
    assert train_split + val_split <= 0.9, "train_split + val_split must be between 0.1 and 0.9"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_loader = ShakespearePlaysLoader(dataset_dir, level=level)
    
    if level == 'word':
        word2vec_model = data_loader.wv
        embedding_size = data_loader.embedding_dim
    else: 
        word2vec_model = None
        embedding_size = embedding_size

    
    if level == 'word':
        text_word_ids, embedding_matrix, vocab_size = data_loader.preprocess_word_level()
        dataset = data_loader.create_dataset(text_word_ids[:dataset_length])
        token_to_id = data_loader.word_to_id
        id_to_token = data_loader.id_to_word
    else:
        text_char_ids, vocab_size = data_loader.preprocess_char_level()
        dataset = data_loader.create_dataset(text_char_ids[:dataset_length])
        token_to_id = data_loader.char_to_id
        id_to_token = data_loader.id_to_char

    train_loader, val_loader, test_loader = data_loader.create_loaders(dataset, train_split, val_split, batch_size)

    print("data loaded")

    # Initialize the model
    model = None
    embedding_matrix = None
    if model_name == 'RNN':
        model = RNNModel(input_size, hidden_size, vocab_size, num_layers, embedding_size, embedding_matrix)
    else:
        model = LSTMModel(input_size, hidden_size, vocab_size, num_layers, embedding_size, embedding_matrix)

    # Move the model to GPU if available
    model.to(device)

    # Define the loss function, learning rate, and optimizer
    if level == 'word':
        criterion = nn.CrossEntropyLoss(ignore_index=token_to_id['<PAD>'])  # Ignore padding tokens
    else:
        criterion = nn.CrossEntropyLoss()  # Ignore padding tokens

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if model_name == 'LSTM':
        trainer = ModelTrainer(model, train_loader, criterion, optimizer, device, number_states=2)
    else:
        trainer = ModelTrainer(model, train_loader, criterion, optimizer, device, number_states=1)

    # Training loop
    for epoch in range(num_epochs):
        print("starting epoch", epoch+1)
        loss = trainer.train()
        validation_loss = trainer.evaluate(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Validation Loss: {validation_loss:.4f}")

    ## Test the model
    test_loss = trainer.evaluate(test_loader)
    print("test loss:", test_loss)

    if model_name == 'LSTM':
        evaluater = Evaluater(model, device, number_states=2)
    else:
        evaluater = Evaluater(model, device, number_states=1)

    return evaluater, token_to_id, id_to_token, test_loader, criterion, device

if __name__ == '__main__':
    # Hyperparameters
    input_size = 100
    hidden_size = 256
    embedding_size = 100
    num_layers = 1
    batch_size = 50
    train_split = 0.8
    val_split = 0.1
    num_epochs = 3
    learning_rate = 0.001
    dataset_length = -1
    dataset_dir = './data/ShakespearePlays/'
    level = 'word'
    model_name = 'RNN'

    evaluater, word_to_id, id_to_word, test_loader, criterion, device = train(dataset_dir, level=level, 
                                                                            model_name=model_name,
                                                                            batch_size=batch_size, num_epochs=num_epochs,
                                                                            train_split=0.8, val_split=0.1, 
                                                                            learning_rate=learning_rate, 
                                                                            hidden_size=hidden_size, 
                                                                            embedding_size=embedding_size, 
                                                                            num_layers=num_layers, input_size=input_size, 
                                                                            dataset_length=dataset_length, use_bpe=False, 
                                                                            use_augmentation=False)
    
    perplexity = evaluater.calculate_perplexity(test_loader, criterion)
    print('Perplexity:', perplexity)

    seed_text = "Start a discussion about coffee"
    gen_length = 2000

    generated_text = evaluater.generate_text(seed_text, gen_length, word_to_id, id_to_word, level, device, temperature=1, top_p=0)
    print(generated_text)



# data_processor_char = ShakespearePlaysLoader(DATASET_DIR, level='char')

# word2vec_model = data_loader.wv
# EMBEDDING_SIZE = data_loader.embedding_dim

# text_word_ids, word_to_id, id_to_word, embedding_matrix, vocab_size = data_loader.preprocess()
# dataset = data_loader.create_dataset(text_word_ids[:10000])
# train_loader, val_loader, test_loader = data_loader.create_loaders(dataset, 0.8, 0.1, 0.1, 50)

# print("data loaded")

# # Initialize the LSTM model with an embedding layer
# model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, vocab_size, NUM_LAYERS, EMBEDDING_SIZE, embedding_matrix)

# # Move the model to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Define the loss function, learning rate, and optimizer
# criterion = nn.CrossEntropyLoss(ignore_index=word_to_id['<PAD>'])  # Ignore padding tokens
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# trainer = ModelTrainer(model, train_loader, criterion, optimizer, device)

# # Training loop
# for epoch in range(NUM_EPOCHS):
#     print("starting epoch", epoch+1)
#     loss = trainer.train()
#     validation_loss = trainer.evaluate(val_loader)
#     print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss:.4f}, Validation Loss: {validation_loss:.4f}")

# ## Evaluate the model

# test_loss = trainer.evaluate(test_loader)

# evaluater = Evaluater(model, device)

# perplexity = evaluater.calculate_perplexity(test_loader, criterion)
# print('Perplexity:', perplexity)

# seed_text = "Start a discussion about coffee"
# gen_length = 2000

# generated_text = evaluater.generate_text(seed_text, gen_length, word_to_id, id_to_word, device, temperature=1, top_p=0)
# print(generated_text)