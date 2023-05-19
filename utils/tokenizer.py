from tokenizers import Tokenizer
from tokenizers.models import WordLevel, BPE
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
import sys
sys.path.append('./')
from data_loader.data_loaders import ShakespearePlaysLoader

class BPETokenizer:
    def __init__(self, vocab_size=30000, min_frequency=2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        self.tokenizer.pre_tokenizer = Whitespace()
        
    def train(self, file_name):
        print("Training tokenizer ...")
        trainer = BpeTrainer(vocab_size=self.vocab_size, min_frequency=self.min_frequency, special_tokens=["<PAD>", "<UNK>"])
        self.tokenizer.train([file_name], trainer)
        print("Tokenizer trained.")

    def save(self, path):
        print("Saving tokenizer ...")
        self.tokenizer.save(path)
        print("Tokenizer saved.")

    def load(self, path):
        print("Loading tokenizer ...")
        self.tokenizer = Tokenizer.from_file(path)
        print("Tokenizer loaded.")

    def use_pretrained(self, pretrained_name):
        print("Changing to pretrained tokenizer ...")
        self.tokenizer = Tokenizer.from_pretrained(pretrained_name)


# thoroughtly test the tokenizer
if __name__ == '__main__':
    # load the data
    shakes_file = './data/ShakespearePlays/shakespeare.txt'
    loader = ShakespearePlaysLoader(shakes_file, level='word')
    plays = loader.read_Plays()
    word_to_id, id_to_word = loader.build_word_vocab()
    vocab_size = len(word_to_id)
    print("vocab size:", vocab_size)

    tokenizer = BPETokenizer(vocab_size=vocab_size, min_frequency=0)
    # tokenizer.train(shakes_file)
    # tokenizer.save('./data/tokenizers/bpe_tokenizer.json')
    tokenizer.load('./data/tokenizers/bpe_tokenizer.json')
    # get the mapping for the file using the tokenizer
    encoded = tokenizer.tokenizer.encode(plays)
    print("vocab size:", len(encoded.tokens))


