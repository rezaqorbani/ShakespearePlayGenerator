import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from data_loader.ShakespearePlaysLoader import ShakespearePlaysLoader
from models.models import LSTMModel, RNNModel
from trainers.trainer import ModelTrainer
import torch.nn as nn
from tqdm import tqdm
from utils.augmenter import TextAugmenter
from test import Evaluater
import os
import json
import matplotlib.pyplot as plt

def train(loader, dataset_dir, level='char', model_name='RNN', batch_size=64, train_split=0.8, val_split=0.1, num_epochs=2,
           learning_rate=0.001, hidden_size=256, embedding_size=100, num_layers=1, input_size=100, dataset_length=1000, 
            use_augmentation=False, early_stopping_patience=0, gamma=0.9, step_size=1, use_scheduler=False):
    assert train_split + val_split <= 0.9, "train_split + val_split must be between 0.1 and 0.9"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = loader(dataset_dir, level=level, augment=use_augmentation)
    last_epoch = 0
    print("training started...")
    print("loading data...")
    if level == 'word':
        word2vec_model = data_loader.wv
        embedding_size = data_loader.embedding_dim
    else: 
        word2vec_model = None
        embedding_size = embedding_size

    
    if level == 'bpe':
        text_bpe_ids, vocab_size = data_loader.preprocess_bpe_level()
        dataset = data_loader.create_dataset(text_bpe_ids[:dataset_length])
        token_to_id = data_loader.token_to_id
        id_to_token = data_loader.id_to_token
        tokenizer = data_loader.tokenizer
    elif level == 'word':
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
    if os.path.exists(f'./saved/models/{model_name}_{level}_{batch_size}_{learning_rate}_{hidden_size}_{embedding_size}_{num_layers}_model.pt'):
        print("loading model...")
        checkpoint = torch.load(f'./saved/models/{model_name}_{level}_{batch_size}_{learning_rate}_{hidden_size}_{embedding_size}_{num_layers}_model.pt')
        model = LSTMModel if model_name == 'LSTM' else RNNModel
        model = model(input_size, hidden_size, vocab_size, num_layers, embedding_size, embedding_matrix)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['last_epoch']
        loss = checkpoint['loss']
        print("model loaded")
    else:
        if model_name == 'RNN':
            model = RNNModel(input_size, hidden_size, vocab_size, num_layers, embedding_size, embedding_matrix)
        else:
            model = LSTMModel(input_size, hidden_size, vocab_size, num_layers, embedding_size, embedding_matrix)

    # Move the model to GPU if available
    model.to(device)

    # Define the loss function, learning rate, and optimizer
    if level == 'word' or level == 'bpe':
        criterion = nn.CrossEntropyLoss(ignore_index=token_to_id['<PAD>'])  # Ignore padding tokens
    else:
        criterion = nn.CrossEntropyLoss()  # Ignore padding tokens

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    schedular = StepLR(optimizer, step_size=step_size, gamma=gamma)

    if model_name == 'LSTM':
        trainer = ModelTrainer(model, train_loader, criterion, optimizer, device, number_states=2)
    else:
        trainer = ModelTrainer(model, train_loader, criterion, optimizer, device, number_states=1)

    # Training loop
    SAVE_AFTER_EPOCHS = 2
    SAVE_PATH = f'./saved/models/{model_name}_{level}_{batch_size}_{learning_rate}_{hidden_size}_{embedding_size}_{num_layers}_model.pt'

    if early_stopping_patience > 0:
        early_stopping_counter = 0
        best_loss = float('inf')

    val_losses = []
    train_losses = []
    for epoch in range(last_epoch, num_epochs):
        print("starting epoch", epoch+1)
        train_loss = trainer.train()
        train_losses.append(train_loss)
        validation_loss = trainer.evaluate(val_loader)
        val_losses.append(validation_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}")
        if epoch % SAVE_AFTER_EPOCHS == 0 and epoch != 0:
            torch.save({
                'last_epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                }, SAVE_PATH)
            
            print("model saved")

        if early_stopping_patience > 0:
            if validation_loss < best_loss:
                best_loss = validation_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter == early_stopping_patience:
                    print("early stopping")

                    test_loss = trainer.evaluate(test_loader)
                    print("test loss:", test_loss)

                    if model_name == 'LSTM':
                        evaluater = Evaluater(model, device, number_states=2)
                    else:
                        evaluater = Evaluater(model, device, number_states=1)

                    return train_losses, val_losses, test_loss, evaluater, token_to_id, id_to_token, test_loader, criterion, device, tokenizer if level == 'bpe' else None

        if use_scheduler:
            schedular.step()
            

    print("training finished")


    ## Test the model
    test_loss = trainer.evaluate(test_loader)
    print("test loss:", test_loss)

    if model_name == 'LSTM':
        evaluater = Evaluater(model, device, number_states=2)
    else:
        evaluater = Evaluater(model, device, number_states=1)

    return train_losses, val_losses, test_loss, evaluater, token_to_id, id_to_token, test_loader, criterion, device, tokenizer if level == 'bpe' else None

if __name__ == '__main__':
    # Hyperparameters
    input_size = 100
    hidden_size = 256
    embedding_size = 100
    num_layers = 2
    batch_size = 101
    train_split = 0.8
    val_split = 0.1
    num_epochs = 30
    learning_rate = 0.001
    dataset_length = -1
    dataset_dir = './data/ShakespearePlays'
    loader = ShakespearePlaysLoader
    level = 'bpe'
    model_name = 'LSTM'
    early_stopping_patience = 3 # set zero to disable early stopping
    step_size = 1
    gamma = 0.9
    use_scheduler = False
    use_augmentation = False

    train_losses, val_losses, test_loss, evaluater, token_to_id, id_to_token, test_loader, criterion, device, tokenizer = train(loader, dataset_dir, level=level, 
                                                                            model_name=model_name,
                                                                            batch_size=batch_size, num_epochs=num_epochs,
                                                                            train_split=train_split, val_split=val_split, 
                                                                            learning_rate=learning_rate, 
                                                                            hidden_size=hidden_size, 
                                                                            embedding_size=embedding_size, 
                                                                            num_layers=num_layers, input_size=input_size, 
                                                                            dataset_length=dataset_length, 
                                                                            use_augmentation=use_augmentation, 
                                                                            early_stopping_patience=early_stopping_patience, 
                                                                            step_size=step_size, gamma=gamma,
                                                                            use_scheduler=use_scheduler)
    
    perplexity = evaluater.calculate_perplexity(test_loader, criterion)
    print('Perplexity:', perplexity)

    seed_text = "ROMEO:"
    gen_length = 300

    generated_text = evaluater.generate_text(seed_text, gen_length, token_to_id, id_to_token, level, device, temperature=0.9, top_p=0.9, tokenizer=tokenizer)
    print(generated_text)

    # save the plot of the validation losses and train losses in the results/plots folder. The plot should have a legend (train, validation) and a title (Model, tokenization level), shift the epochs by 1
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{model_name}, {level}')
    plt.savefig(f'./results/plots/{model_name}_{level}_{batch_size}_{learning_rate}_{hidden_size}_{embedding_size}_{num_layers}_plot.png')
    plt.close()

    # append the perplexity, test_loss and generated text in a text file in the results/generation folder, create the file if it doesn't exist
    with open(f'./results/generation/{model_name}_{level}_{batch_size}_{learning_rate}_{hidden_size}_{embedding_size}_{num_layers}_generation.txt', 'a+') as f:
        f.write(f'Perplexity: {perplexity}\n')
        f.write(f'Test Loss: {test_loss}\n')
        f.write(f'Generated Text:\n{generated_text}\n')
        f.write('----------------------------------------\n\n')

