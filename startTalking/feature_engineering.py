# preprocess.py
## Character-Level Mapping

import os
import csv
import json

class DataProcessor:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

    def read_dialogue_lines(self):
        dialogue_lines = {}
        with open(os.path.join(self.dataset_dir, 'movie_lines.txt'), 'r', encoding='iso-8859-1') as f:
            for row in f:
                line=row.strip().split(" +++$+++ ")
                if len(line) == 5 and line[4]:
                    dialogue_lines[line[0]] = line[4].strip()
                
        return dialogue_lines

    def build_char_vocab(self, text):
        unique_chars = sorted(set(text))
        char_to_id = {char: idx for idx, char in enumerate(unique_chars)}
        id_to_char = {idx: char for idx, char in enumerate(unique_chars)}
        char_to_id['<UNK>'] = len(unique_chars)
        id_to_char[len(unique_chars)] = '<UNK>'
        return char_to_id, id_to_char

    def text_to_ids(self, text, char_to_id):
        ids = [char_to_id.get(char, char_to_id['<UNK>']) for char in text]
        return ids

    def preprocess(self):
        dialogue_lines = self.read_dialogue_lines()
        text = ' '.join(dialogue_lines.values())
        
        char_to_id, id_to_char = self.build_char_vocab(text)
        ids = self.text_to_ids(text, char_to_id)

        return ids, char_to_id, id_to_char

    def save_mappings(self, char_to_id_file, char_to_id, id_to_char_file, id_to_char):
        with open(char_to_id_file, 'w') as f:
            json.dump(char_to_id, f)

        with open(id_to_char_file, 'w') as f:
            json.dump(id_to_char, f)

    def load_mappings(self, char_to_id_file, id_to_char_file):
        with open(char_to_id_file, 'r') as f:
            char_to_id = json.load(f)

        with open(id_to_char_file, 'r') as f:
            id_to_char = json.load(f)
        return char_to_id, id_to_char
