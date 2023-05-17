import os
import gensim
from gensim.models import Word2Vec

class Word2VecTrainer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.n_words = self.word_count
        
    def read_dialogue_lines(self):
        dialogue_lines = []
        with open(os.path.join(self.dataset), 'r', encoding='iso-8859-1') as f:
            for row in f:
                line=row.strip().split(" +++$+++ ")
                if len(line) == 5 and line[4]:
                    dialogue_lines.append(line[4].strip().split())
        return dialogue_lines
    
    def word_count(self, dataset):
        word_count = {}
        for line in dialogue_lines:
            for word in line:
                if word not in word_count:
                    word_count[word] = 0
                word_count[word] += 1
        return word_count
    
    def train_word2vec(self, sentences, min_count=5, size=100, window=5, workers=4):
        model = Word2Vec(sentences, min_count=min_count, vector_size=size, window=window, workers=workers)
        return model

    def save_word2vec(self, model, path):
        model.save(path)

if __name__ == '__main__':

    dataset= './data/Cornell_Movie-Dialog_Corpus/movie_lines.txt'
    word2vec_path = './data/embeddings/word2vec.model'

    trainer = Word2VecTrainer(dataset)
    dialogue_lines = trainer.read_dialogue_lines()
    model = trainer.train_word2vec(dialogue_lines)
    trainer.save_word2vec(model, word2vec_path)
