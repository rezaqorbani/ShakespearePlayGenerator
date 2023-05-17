import os
import json
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from gensim.models import Word2Vec

class CornellMovieDialogsLoader:
    def __init__(self, dataset_dir, level='word'):
        self.dataset_dir = dataset_dir
        self.word2vec_path = './data/embeddings/word2vec.model'
        self.word2vec = Word2Vec.load(self.word2vec_path)
        self.embedding_dim = self.word2vec.vector_size
        self.level = level
        self.vocab_size = None
        self.mappings_dir = './data/mappings'

    def read_dialogue_lines(self):
        dialogue_lines = {}
        with open(os.path.join(self.dataset_dir, 'movie_lines.txt'), 'r', encoding='iso-8859-1') as f:
            for row in f:
                line=row.strip().split(" +++$+++ ")
                if len(line) == 5 and line[4]:
                    dialogue_lines[line[0]] = line[4].strip()
               
        return dialogue_lines
    
    def build_word_vocab(self):
        # check if the mapping folder is empty, then save the mappings, else load the mappings
        if not os.listdir('./data/mappings'):
            # Add special tokens
            self.word_to_id = {'<PAD>': 0, '<UNK>': 1}
            self.id_to_word = {0: '<PAD>', 1: '<UNK>'}

            # Add words from word2vec model
            for word in self.word2vec.wv.key_to_index:
                if word not in self.word_to_id:
                    idx = len(self.word_to_id)
                    self.word_to_id[word] = idx
                    self.id_to_word[idx] = word
            
            self.vocab_size = len(self.word_to_id)
            self.save_mappings(self.mappings_dir, self.word_to_id, self.id_to_word)
        else:
            self.word_to_id, self.id_to_word = self.load_mappings(self.mappings_dir)
        
        self.vocab_size = len(self.word_to_id)

        return self.word_to_id, self.id_to_word

    def text_to_ids(self, dialog_lines, word_to_id):
        ids = [[word_to_id.get(word, word_to_id['<UNK>']) for word in line] for line in dialog_lines.values()]
        # text_ids = [word_to_id.get(word, word_to_id['<UNK>']) for word in text]
        return ids
    
    def text_to_vectors(self, text):
        vectors = [self.word2vec.wv[word] if word in self.word2vec.wv else self.word2vec.wv['<UNK>'] for word in text]
        return vectors

    def preprocess(self):
        dialogue_lines = self.read_dialogue_lines()

        word_to_id, id_to_word = self.build_word_vocab()
        text_id = self.text_to_ids(dialogue_lines, word_to_id)

        # Create embedding matrix
        embedding_matrix = torch.zeros((len(word_to_id), self.embedding_dim))
        for word, idx in word_to_id.items():
            if word in self.word2vec.wv:
                embedding_matrix[idx] = torch.tensor(self.word2vec.wv[word])
        
        
        return text_id, word_to_id, id_to_word, embedding_matrix, self.vocab_size
    
    def create_dataset(self, tokenized_text, seq_length=50):
        input_sequences = []
        target_sequences = []

        # Iterate over each sentence in tokenized_text
        for sentence in tokenized_text:
            # Create input-target pairs with the specified sequence length
            for i in range(0, len(sentence) - seq_length, 1):
                input_seq = sentence[i:i + seq_length]
                target_seq = sentence[i + 1:i + seq_length + 1]
                input_sequences.append(input_seq)
                target_sequences.append(target_seq)

        # Padding input sequences and target sequences
        input_sequences = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in input_sequences], batch_first=True)
        target_sequences = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in target_sequences], batch_first=True)

        dataset = TensorDataset(input_sequences, target_sequences)
        return dataset

    
    @staticmethod
    def create_loaders(dataset, train_split, val_split, test_split, batch_size):
      train_size = int(train_split * len(dataset))
      val_size = int(val_split * len(dataset))
      test_size = len(dataset) - train_size - val_size  # this accounts for any remainder
      train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
      train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
      val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
      test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
      return train_loader, val_loader, test_loader
    
    def save_mappings(self, save_dir, word_to_id, id_to_word):
        with open(os.path.join(save_dir, f'{self.level}_to_id.json'), 'w') as f:
            json.dump(word_to_id, f)

        with open(os.path.join(save_dir, f'id_to_{self.level}.json'), 'w') as f:
            json.dump(id_to_word, f)
    
    def load_mappings(self, load_dir):
        with open(os.path.join(load_dir, f'{self.level}_to_id.json'), 'r') as f:
            word_to_id = json.load(f)

        with open(os.path.join(load_dir, f'id_to_{self.level}.json'), 'r') as f:
            id_to_word = json.load(f)

        return word_to_id, id_to_word