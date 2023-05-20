import os
import json
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api
from tqdm import tqdm
from utils.tokenizer import BPETokenizer
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize


class ShakespearePlaysLoader:
    def __init__(self, dataset_dir, level='word'):
        self.dataset_dir = dataset_dir
        if level == 'word':
          self.word2vec_path = './data/embeddings/word2vec.wordvectors'
          self.wv = KeyedVectors.load(self.word2vec_path, mmap='r')
          # self.wv = api.load('word2vec-google-news-300')
          # self.wv = KeyedVectors.load_word2vec_format('
          self.embedding_dim = self.wv.vector_size
        self.level = level
        self.vocab_size = None
        self.mappings_dir = dataset_dir+'/mappings'
        self.plays_data= self.read_Plays()
        self.tokenized_plays = self.tokenize(self.plays_data)

    def read_Plays(self):
        print('Reading Shakespeare Plays...')
        with open(self.dataset_dir+"/Plays/shakespeare.txt", 'r') as f:
            plays_data = f.read()      
        return plays_data

    def tokenize(self, text):
        text = text.replace("\n", " NewLine ")
        text = word_tokenize(text)
        text = [token.replace("NewLine", "\n") for token in text]
        return text

    def build_word_vocab(self):
        print('Building word vocabulary...')
        # check if the mapping folder is empty, then save the mappings, else load the mappings
        # if not os.listdir('./data/mappings'):
        if os.path.exists(self.mappings_dir + f'/{self.level}_to_id.json') and os.path.exists(self.mappings_dir + f'/id_to_{self.level}.json'):
            self.word_to_id, self.id_to_word = self.load_mappings(self.mappings_dir)
        else:
            # Add special tokens
            self.word_to_id = {'<PAD>': 0, '<UNK>': 1}
            self.id_to_word = {0: '<PAD>', 1: '<UNK>'}

            # Add words from word2vec model
            for word in tqdm(self.tokenized_plays):
            #for word in self.wv.key_to_index:
                if word not in self.word_to_id:
                    idx = len(self.word_to_id)
                    self.word_to_id[word] = idx
                    self.id_to_word[idx] = word
            
            self.vocab_size = len(self.word_to_id)
            self.save_mappings(self.mappings_dir, self.word_to_id, self.id_to_word)
        
        self.vocab_size = len(self.word_to_id)

        return self.word_to_id, self.id_to_word
    
    def build_char_vocab(self):
        print('Building character vocabulary...')
        # check if the mapping folder is empty, then save the mappings, else load the mappings
        if os.path.exists(self.mappings_dir + f'/{self.level}_to_id.json') and os.path.exists(self.mappings_dir + f'/id_to_{self.level}.json'):
            self.char_to_id, self.id_to_char = self.load_mappings(self.mappings_dir)
        else:
            unique_chars = sorted(set(self.plays_data))
            self.char_to_id = {char: idx for idx, char in enumerate(unique_chars)}
            self.id_to_char = {idx: char for idx, char in enumerate(unique_chars)}
            self.char_to_id['<UNK>'] = len(unique_chars)
            self.id_to_char[len(unique_chars)] = '<UNK>'
            self.save_mappings(self.mappings_dir, self.char_to_id, self.id_to_char)
            
        self.vocab_size = len(self.char_to_id)
        return self.char_to_id, self.id_to_char
    
    def build_bpe_vocab(self):
        print('Building BPE vocabulary...')
        # check if the mapping folder is empty, then save the mappings, else load the mappings
        self.tokenizer = BPETokenizer()
        # if os.path.exists(f'./data/mappings/{self.level}/bpe_tokenizer.json'):
        #     self.tokenizer.load('./data/mappings/bpe/bpe_tokenizer.json')
        #     self.vocab_size = self.tokenizer.vocab_size
        # else:
        #     # Add special tokens
        plays_path = './data/ShakespearePlays/shakespeare.txt'
        self.tokenizer.train(plays_path)
        # self.tokenizer.save('./data/mappings/bpe/bpe_tokenizer.json')
        self.vocab_size = self.tokenizer.vocab_size

        self.token_to_id = self.tokenizer.tokenizer.get_vocab()
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        return self.token_to_id, self.id_to_token      
        
    def text_to_ids_word(self, plays, word_to_id):
        ids = [word_to_id.get(word, word_to_id['<UNK>']) for word in plays]
        return ids
    
    def text_to_ids_char(self, plays, char_to_id):
        ids = [char_to_id.get(char, char_to_id['<UNK>']) for char in plays]
        return ids
    
    def text_to_ids_bpe(self, plays, tokenizer):
        ids = tokenizer.tokenizer.encode(plays).ids
        return ids
    
    def preprocess_word_level(self):
        print('Preprocessing data...')
        word_to_id, id_to_word = self.build_word_vocab()
        text_id = self.text_to_ids_word(self.tokenized_plays, word_to_id)

        # Create embedding matrix
        embedding_matrix = torch.zeros((len(word_to_id), self.embedding_dim))
        for word, idx in word_to_id.items():
            if word in self.wv:
                embedding_matrix[idx] = torch.tensor(self.wv[word])
             
        return text_id, embedding_matrix, self.vocab_size
    
    def preprocess_char_level(self):
        print('Preprocessing data...')
        char_to_id, id_to_char = self.build_char_vocab()
        text_id = self.text_to_ids_char(self.plays_data, char_to_id)
        return text_id, self.vocab_size
    
    def preprocess_bpe_level(self):
        print('Preprocessing data...')
        token_to_id, id_to_token = self.build_bpe_vocab()
        text_id = self.text_to_ids_bpe(self.plays_data, self.tokenizer)
        return text_id, self.vocab_size
    
    def create_dataset(self, tokenized_text, seq_length=50):
        print('Creating dataset...')
        input_sequences = []
        target_sequences = []

        # Create input-target pairs with the specified sequence length
        for i in tqdm(range(0, len(tokenized_text) - seq_length, 1)):
            input_seq = tokenized_text[i:i + seq_length]
            target_seq = tokenized_text[i + 1:i + seq_length + 1]
            input_sequences.append(input_seq)
            target_sequences.append(target_seq)

        dataset = TensorDataset(torch.tensor(input_sequences, dtype=torch.long),
                                torch.tensor(target_sequences, dtype=torch.long))
        return dataset


    def extract_passages(self):
        text = self.plays_data.split('\n\n')
        passages = [re.sub(r'^.*?:', '', line).strip() for line in text if re.sub(r'^.*?:', '', line).strip() != '']
        characters = [re.findall(r'^.*?:', line)[0] for line in text if re.sub(r'^.*?:', '', line).strip() != '']
        return passages, characters

    
    @staticmethod
    def create_loaders(dataset, train_split, val_split, batch_size):
      train_size = int(train_split * len(dataset))
      val_size = int(val_split * len(dataset))
      test_size = len(dataset) - train_size - val_size  # this accounts for any remainder
      train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
      train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
      val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
      test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
      return train_loader, val_loader, test_loader
    
    def save_mappings(self, save_dir, token_to_id, id_to_token):
        with open(os.path.join(save_dir, f'{self.level}_to_id.json'), 'w') as f:
            json.dump(token_to_id, f) 

        with open(os.path.join(save_dir, f'id_to_{self.level}.json'), 'w') as f:
            json.dump(id_to_token, f)
    
    def load_mappings(self, load_dir):
        with open(os.path.join(load_dir, f'{self.level}_to_id.json'), 'r') as f:
            data = json.load(f)
            token_to_id = {k: int(v) for k, v in data.items()}

        with open(os.path.join(load_dir, f'id_to_{self.level}.json'), 'r') as f:
            data = json.load(f)
            id_to_token = {int(k): v for k, v in data.items()}


        return token_to_id, id_to_token