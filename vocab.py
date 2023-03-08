from collections import Counter
import pickle
import os

class Vocabulary:

    def __init__(self, vocab_size):
        """
        TODO: Change vocabulary code as you need. (e.g. tokenizer, using stopword, etc.)
        
        vocab_size : max source vocab size. Eg. if set to 10,000, we pick the top 10,000 most frequent words and discard others
        """

        ## <PAD> -> padding, used for padding the shorter sentences in a batch to match the length of longest sentence in the batch
        ## <UNK> -> words which are not found in the vocab are replace by this token
        self.vocab_size = vocab_size
        self.itow = {0: '<PAD>',1: '<UNK>'}
        self.wtoi = {w:i for i, w in self.itow.items()}


    def __len__(self):
        return len(self.itow)

    def tokenizer(self, text):
        return [tok.lower().strip() for tok in text.split(' ')]

    def build_vocabulary(self, sentences, add_unk=True):
        word_list = []
        idx=2 #index from which we want our dict to start. We already used 2 indexes for pad and unk

        if add_unk == False: # build vocab for label
            self.wtoi = {}
            idx=0
        
        for sentence in sentences:
            for word in self.tokenizer(sentence):
                if word:
                    word_list.append(word)
                    
        freq = Counter(word_list)
        freq_ = sorted(freq,key=freq.get,reverse=True)

        for word in freq_[:self.vocab_size-idx]:
            self.wtoi[word] = idx
            self.itow[idx] = word
            idx += 1


    def sentence_to_numeric(self, text):
        tokenized_text = self.tokenizer(text)
        numeric_text = []
        for token in tokenized_text:
            if token in self.wtoi:
                numeric_text.append(self.wtoi[token])
            else: # out-of-vocab words are replaced by <UNK>
                numeric_text.append(self.wtoi['<UNK>'])

        return numeric_text


    """
    Save and Load Vocabulary
    """

    # Use in train
    def save_vocabulary(self, name):
        path = './pickles'
        if not os.path.exists(path):
            os.mkdir(path)

        with open(f'{path}/{name}_itow.pkl', 'wb') as w:
            pickle.dump(self.itow, w)
        
        with open(f'{path}/{name}_wtoi.pkl', 'wb') as w:
            pickle.dump(self.wtoi, w)

    # Use in test
    def load_vocabulary(self, name, path):
        with open(f'{path}/{name}_itow.pkl', 'rb') as f:
            self.itow = pickle.load(f)

        with open(f'{path}/{name}_wtoi.pkl', 'rb') as f:
            self.wtoi = pickle.load(f)
            