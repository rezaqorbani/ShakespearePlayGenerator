from gensim.models import Word2Vec
import sys
import os
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
nltk.download('punkt')
sys.path.insert(0, os.path.abspath('.'))
from data_loader import ShakespearePlaysLoader

class Word2VecModel:
    def __init__(self, dataset_dir, loader, embedding_size):
        self.dataset_dir = dataset_dir
        self.loader = loader
        self.embedding_size = embedding_size
        self.model=None


        self.plays_tokenized = self.loader(self.dataset_dir, 'word').tokenized_plays

        self.plays=self.loader(self.dataset_dir, 'word').plays_data.replace('\n\n', ' NewLine ')
        
        # Sentence splitting
        self.sentences = sent_tokenize(self.plays)

        # Word tokenization
        self.sentences = [word_tokenize(sentence) for sentence in self.sentences]
        
        
    def train_word2vec(self, min_count=1, size=300, window=5, workers=8):
        self.model = Word2Vec(self.sentences, min_count=min_count, vector_size=size, window=window, workers=workers)
        
 
    def save_model(self, path):
        self.model.save(path)
        
    def save_embeddings(self, path):
        self.model.wv.save_word2vec_format(path, binary=False)
    
    def load_embeddings(self, filename):
        embeddings = {}
        with open(filename, 'r') as f:
            first_line = f.readline().split()
            self.vocab_size = int(first_line[0])
            self.embedding_size = int(first_line[1])
            for line in f:
                values = line.split()
                word = values[0].replace('NewLine', r'\n')
                vector = [float(x) for x in values[1:]]
                embeddings[word] = vector
        os.remove(filename)

        return embeddings
    
    
    def save_embeddingswith_NL(self, filename, embeddings):
        with open(filename, 'w') as f:
            f.write(str(self.vocab_size) + ' '+ str(self.embedding_size) + '\n')
            for word, vector in embeddings.items():
                f.write(word + ' ' + ' '.join(map(str, vector)) + '\n')

if __name__ == '__main__':
    dataset_dir = './data/ShakespearePlays'
    loader = ShakespearePlaysLoader.ShakespearePlaysLoader
    word2vecModel=Word2VecModel(dataset_dir, loader, embedding_size=300)
    # Train word2vec on our corpus
    word2vecModel.train_word2vec()
    word2vecModel.save_model('./saved/models/pretrained_w2v.model')
    word2vecModel.save_embeddings('./data/embeddings/word2vec_vectors.txt')
    
    embeddings=word2vecModel.load_embeddings('./data/embeddings/word2vec_vectors.txt')
    word2vecModel.save_embeddingswith_NL('./data/embeddings/word2vec_embeddings.txt', embeddings)
    