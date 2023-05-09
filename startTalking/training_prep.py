import torch

def create_training_sequences(text):
    '''Divide text in sequences'''
    sequences = []
    seq_length = 100
    for i in range(0, len(text) - seq_length,seq_length):
        sequences.append(text[i:i + seq_length + 1])
    return sequences

def get_input_target_sequences(sequences, char_to_id):
    '''Divide sequences into target and input'''
    input_sequences = []
    target_sequences = []
    for seq in sequences:
        input_sequences.append([char_to_id[char] for char in seq[:-1]])
        target_sequences.append([char_to_id[char] for char in seq[1:]])
    return input_sequences, target_sequences

def get_dataloader(input_sequences, target_sequences, batch_size):
    '''shuffle and batch the sequences'''
    input_sequences = torch.Tensor(input_sequences)
    target_sequences = torch.Tensor(target_sequences)
    dataset = torch.utils.data.TensorDataset(input_sequences, target_sequences)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def create_data_sequences(text,char_to_id,batch_size):
    '''Create sequences for training'''
    sequences = create_training_sequences(text)
    input_sequences, target_sequences = get_input_target_sequences(sequences, char_to_id)
    dataloader = get_dataloader(input_sequences, target_sequences, batch_size)
    return dataloader