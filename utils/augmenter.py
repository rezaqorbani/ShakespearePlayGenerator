import nlpaug.augmenter.word as naw
from tqdm import tqdm
from data_loader.CornellMovieLoader import CornellMovieLoader


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
    
    def augment(self, data_loader, filename):
        passages, characters=data_loader.extract_passages()
        characters=[character[0]+'\n' if character!=[] else '\n\n' for character in characters ]
        N=6000
        for i in range(N):
            text_context = self.augment_context(passages[i*500:(i+1)*500])
            text = '\n\n'.join([character+elt[0] for character, elt in zip(characters[i*500:(i+1)*500],text_context)])
            # Write the text to a file
            if i==0:
                with open(filename, 'w') as f:
                    f.write(text)
            else:
                with open(filename, 'a') as f:
                    f.write('\n\n'+text)
                    
if __name__ == '__main__':
    dataset_dir= './data/ShakespearePlays'
    p=0.8
    textAugmenter = TextAugmenter(augmentation_percentage=p)
    data_loader = CornellMovieLoader(dataset_dir)
    
    textAugmenter.augment(data_loader, './data/ShakespearePlays/augmentation/augmented_text.txt')
