import pickle
import numpy as np
from pyvi import ViTokenizer
from service.TextPreprocessing import *


class MyTokenizer(object):
    tokenizer_custom = None

    @classmethod
    def load_tokenizer(self):
        if self.tokenizer_custom is None:
            with open('./modes/tokenizer.pickle', 'rb') as handle:
                self.tokenizer_custom = pickle.load(handle)
        return self.tokenizer_custom

    def fit_on_text(self, target):
        def tokenize(target):

            return ViTokenizer.tokenize(clean_str(target)).split(' ')

        def buildVector(seq_word, tokenzier_custom):
            seq_vec = np.zeros(100, dtype=np.int64)
            # i=-1
            for i, word in enumerate(seq_word):
                if i >= 100:
                    break;
                if word in tokenzier_custom:
                    seq_vec[i] = tokenzier_custom[word]
                else:
                    i = i - 1
            return seq_vec

        X_train = []
        for i, word in enumerate(target):
            X_train.append(buildVector(tokenize(word), self.tokenizer_custom))
        return np.array(X_train)
