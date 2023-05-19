import os
import gensim
from gensim.models import Word2Vec
import gensim.downloader
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
    
    def train_word2vec(self, sentences, min_count=5, size=100, window=5, workers=8):
        model = Word2Vec(sentences, min_count=min_count, vector_size=size, window=window, workers=workers)
        model.build_vocab(sentences, progress_per=1)
        model.train(sentences, total_examples=model.corpus_count, epochs=30, report_delay=1)
        return model
    
    def load_pretrained_word2vec(self, pretrained_name):
        model = gensim.downloader.load(pretrained_name)
        return model

    def save_word2vec(self, model, path):
        model.wv.save(path)

    def save_pretrained_word2vec(self, model, path):
        model.save(path)

if __name__ == '__main__':
    dataset= './data/Cornell_Movie-Dialog_Corpus/movie_lines.txt'
    word2vec_pretrained_name = 'word2vec-google-news-300'
    word2vec_path = './data/embeddings/word2vec.wordvectors'

    trainer = Word2VecTrainer(dataset)
    pretrained_model = trainer.load_pretrained_word2vec(word2vec_pretrained_name)
    trainer.save_pretrained_word2vec(pretrained_model, word2vec_path)
    # dialogue_lines = trainer.read_dialogue_lines()
    # model = trainer.train_word2vec(dialogue_lines)
    # trainer.save_word2vec(model, word2vec_path)
