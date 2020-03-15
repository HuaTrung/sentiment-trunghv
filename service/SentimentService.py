import pickle
import time

import tensorflow as tf
import numpy as np
from pyvi import ViTokenizer
from service.TextPreprocessing import *
import os



class SentimentService(object):
    model1 = None
    tokenizer_custom = None

    @classmethod
    def load_deep_model(self, model):
        from app import APP_ROOT
        APP_STATIC = os.path.join(APP_ROOT, 'models')
        # Reload the model from the 2 files we saved
        with open(APP_STATIC+"/" + model + "_config.json") as json_file:
            json_config = json_file.read()
        loaded_model = tf.keras.models.model_from_json(json_config)
        loaded_model.load_weights(APP_STATIC+"/" + model + ".h5")
        return loaded_model

    @classmethod
    def get_model1(self):
        # start_time = time.time()
        if self.model1 is None:
            print("loaded module")
            self.model1 = self.load_deep_model('1_2_3_200_filter_100')
        # elapsed_time = time.time() - start_time
        # print('Load model: ' + str(elapsed_time))

        return self.model1

    @classmethod
    def load_tokenizer(self):
        # start_time = time.time()
        from app import APP_ROOT
        APP_STATIC = os.path.join(APP_ROOT, 'models')
        if self.tokenizer_custom is None:
            # print("loaded tokenizer")
            with open(APP_STATIC + '/tokenizer.pickle', 'rb') as handle:
                self.tokenizer_custom = pickle.load(handle)
        # elapsed_time = time.time() - start_time
        # print('Load tokenizer: '+ str(elapsed_time))
        return self

    @classmethod
    def fit_on_text(self, target):
        def tokenize(target):
            return ViTokenizer.tokenize(clean_str(target)).split(' ')

        def buildVector(seq_word, tokenzier_custom):
            seq_vec = np.zeros(100, dtype=np.int64)
            # i=-1
            for i, word in enumerate(seq_word):
                if i >= 100:
                    break
                if word in tokenzier_custom:
                    seq_vec[i] = tokenzier_custom[word]
                else:
                    i = i - 1
            return seq_vec

        X_train = []
        # for i, word in enumerate(target):
        X_train.append(buildVector(tokenize(target), self.tokenizer_custom))
        return np.array(X_train)

    def predict(self, target, tokenizer):
        return self.model1.predict(tokenizer.fit_on_text(target))
