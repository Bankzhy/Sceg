import numpy as np
from sentence_transformers import SentenceTransformer


class DocSim:
    def __init__(self, stopwords=None):
        # self.model = SentenceTransformer(r"D:\all-mini")
        # self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.model = SentenceTransformer('./local-model')
        self.stopwords = stopwords if stopwords is not None else []

    def vectorize(self, doc: str) -> np.ndarray:
        """
        Identify the vector values for each word in the given document
        :param doc:
        :return:
        """
        doc = doc.lower()
        # words = [w for w in doc.split(" ") if w not in self.stopwords]
        # word_vecs = []
        # for word in words:
        #     try:
        #         vec = self.w2v_model[word]
        #         word_vecs.append(vec)
        #     except KeyError:
        #         # Ignore, if the word doesn't exist in the vocabulary
        #         pass
        #
        # # Assuming that document vector is the mean of all the word vectors
        # # PS: There are other & better ways to do it.
        # vector = np.mean(word_vecs, axis=0)
        vector = self.model.encode(doc)
        return vector

    def _cosine_sim(self, vecA, vecB):
        """Find the cosine similarity distance between two vectors."""
        csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
        if np.isnan(np.sum(csim)):
            return 0
        return csim

    def calculate_similarity(self, source_doc, target_docs=None, threshold=0):
        """Calculates & returns similarity scores between given source document & all
        the target documents."""
        # if not target_docs:
        #     return []
        #
        # if isinstance(target_docs, str):
        #     target_docs = [target_docs]

        source_vec = self.vectorize(source_doc)
        target_vec = self.vectorize(target_docs)
        sim_score = self._cosine_sim(source_vec, target_vec)
        # results = []
        # for doc in target_docs:
        #     target_vec = self.vectorize(doc)
        #     sim_score = self._cosine_sim(source_vec, target_vec)
        #     if sim_score > threshold:
        #         results.append({"score": sim_score, "doc": doc})
        #     # Sort results by score in desc order
        #     results.sort(key=lambda k: k["score"], reverse=True)
        sim_score = float(sim_score)
        sim_score = round(sim_score, 2)
        return sim_score
