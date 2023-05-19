import nlpaug.augmenter.word as naw
from tqdm import tqdm


class TextAugmenter:
    def __init__(self, model_type="bert-base-uncased", model_path='GoogleNews-vectors-negative300.bin', augmentation_percentage=0.8):
        self.wordnet_aug = naw.SynonymAug(aug_src='wordnet', aug_p=augmentation_percentage)
        self.context_aug = naw.ContextualWordEmbsAug(
            model_path=model_type, action="substitute", aug_p=augmentation_percentage)
        self.word2vec_aug = naw.WordEmbsAug(
            model_type='word2vec', model_path=model_path, action="substitute", aug_p=augmentation_percentage)

    def augment_wordnet(self, text):
        return [self.wordnet_aug.augment(line) for line in text]

    def augment_context(self, text):
        augmented=[]
        for i in tqdm(range(len(text))):
          augmented.append(self.context_aug.augment(text[i]))
        print('h')
        return augmented


    def augment_word2vec(self, text):
        return [self.word2vec_aug.augment(line) for line in text]
