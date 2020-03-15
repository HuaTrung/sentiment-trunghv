import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import os
class Relevance:
    w2v_model=None
    def __init__(self, word2vec):
        self.w2v_model = word2vec

    @classmethod
    def get_word2vec(self):
        # start_time = time.time()
        from app import APP_ROOT
        APP_STATIC = os.path.join(APP_ROOT, 'models')
        if self.w2v_model is None:
            self.w2v_model = KeyedVectors.load_word2vec_format(APP_STATIC + '/baomoi.window2.vn.model.bin', binary=True)
        # elapsed_time = time.time() - start_time
        # print('Load tokenizer: '+ str(elapsed_time))
        return self.w2v_model

    def vectorize(self, doc: str) -> np.ndarray:
        doc = doc.lower()
        words = [w for w in doc.split(" ")]
        word_vecs = []
        for word in words:
            try:
                vec = self.w2v_model[word]
                word_vecs.append(vec)
            except KeyError:
                pass
        vector = np.mean(word_vecs, axis=0)
        return vector

    def _cosine_sim(self, vecA, vecB):
        """Find the cosine similarity distance between two vectors."""
        relevance = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
        if np.isnan(np.sum(relevance)):
            return 0
        return relevance

    def calculate_similarity(self, source_doc, target_docs=None):
        if not target_docs:
            return {}

        if isinstance(target_docs, str):
            source_vec = self.vectorize(source_doc)
            target_vec = self.vectorize(target_docs)
            return self._cosine_sim(source_vec, target_vec)
        else:
            source_vec = self.vectorize(source_doc)
            results = {}
            for doc in target_docs:
                target_vec = self.vectorize(doc)
                sim_score = self._cosine_sim(source_vec, target_vec)
                results[index]=str(sim_score)
            return results