from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import UnicodeScripts as Pre_tokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import BPEDecoder
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
# from tokenizers.processors import ByteLevel as Post_processor


class BPETokenizer:
    def __init__(self, vocab_size=30000, min_frequency=0):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        self.tokenizer.pre_tokenizer = Pre_tokenizer()
        # self.tokenizer.decoder = BPEDecoder()
        # self.tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
        # self.tokenizer.post_processor = Post_processor(trim_offsets=True)
        
    def train(self, file_name):
        print("Training tokenizer ...")
        trainer = BpeTrainer(vocab_size=self.vocab_size, min_frequency=self.min_frequency, special_tokens=["<PAD>", "<UNK>"])
        self.tokenizer.train([file_name], trainer)
        print("Tokenizer trained.")
        




# thoroughtly test the tokenizer
# if __name__ == '__main__':
    # # load the data
    # shakes_file = 'C:\\Users\\rzaqo\\source\\repos\\StartTalking\\data\\ShakespearePlays\\shakespeare.txt'
    # # # loader = ShakespearePlaysLoader(shakes_file, level='word')
    # # # plays = loader.read_Plays()
    # # # word_to_id, id_to_word = loader.build_word_vocab()
    # # # vocab_size = len(word_to_id)
    # # # print("vocab size:", vocab_size)

    # tokenizer = BPETokenizer(vocab_size=30000, min_frequency=2)
    # file_path = 'C:\\Users\\rzaqo\\source\\repos\\StartTalking\\data\\mappings\\bpe\\bpe_tokenizer.json'
    # tokenizer.train(shakes_file)
    # tokenizer.save(file_path)
    # # tokenizer.load(file_path)
    # # get the mapping for the file using the tokenizer
    # encoded = tokenizer.tokenizer.encode("he was, playing football in the garden, when he fell over. No! he didn't.")
    # decoded = tokenizer.tokenizer.decode(encoded.ids)
    # print("decoded:", decoded)
