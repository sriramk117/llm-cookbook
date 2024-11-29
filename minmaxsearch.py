import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
from scipy.sparse import csr_matrix


# Source code: https://machine-mind-ml.medium.com/discovering-semantic-search-and-rag-with-large-language-models-be7d9ba5bef4 
# reimplementation for my own practice of writing semantic search engines
class Index:
    def __init__(self, text_fields, keyword_fields, vectorizer_params=None):
        self.text_fields = text_fields
        self.keyword_fields = keyword_fields
        self.vectorizers = {field: TfidfVectorizer() for field in text_fields}
        self.keyword_df = None
        self.text_matrices = {}
        self.docs = []

    # processes list of documents and fits the TF-IDF vectorizer to the text fields
    def fit(self, docs):
        self.docs = docs
        keyword_data = {field: [] for field in self.keyword_fields}

        # function to fit a TF-IDF vectorizer to a given text field
        def fit_text_field(field):
            texts = [d.get(field, '') for d in docs]
            return field, self.vectorizers[field].fit_transform(texts)

        # parallelize fitting TF-IDF vectorizers for each text field
        with ThreadPoolExecutor() as executor:
            results = executor.map(fit_text_field, self.text_fields)
            for field, matrix in results:
                self.text_matrices[field] = matrix
        
        # process the odcuments and extract the keyword data
        for doc in docs:
            for field in self.keyword_fields:
                keyword_data[field].append(doc.get(field, ''))

        # store in data frame for filtering later on
        self.keyword_df = pd.DataFrame(keyword_data)
        return self

    # returns top matching documents based on relevance scores (cosine similarity)
    def search(self, query, filter_dict={}, boost_dict={}, num_results = 10):
        # process the query and transform it into TD-IDF vector
        query_vectors = {field: self.vectorizers[field].transform([query]) for field in self.text_fields}
        scores = np.zeros(len(self.docs))

        # compute similarity score given a text field and query vector
        def compute_similarity(field, query_vec):
            # Ensure query_vec is 2D
            if len(query_vec.shape) == 1:
                query_vec = query_vec.reshape(1, -1)
            
            similarity = cosine_similarity(query_vec, self.text_matrices[field]).flatten()
            boost =  boost_dict.get(field, 1) # get the boost value for the field
            return similarity * boost
        
        # parallelize the computation of similarity scores for each text field
        with ThreadPoolExecutor() as executor:
            results = executor.map(lambda field: compute_similarity(field, query_vectors[field]), self.text_fields)
            for result in results:
                scores += result

        # apply filters to the scores
        for field, value in filter_dict.items():
            if field in self.keyword_fields:
                mask = (self.keyword_df[field] == value)
                scores = scores * mask.to_numpy()

        # get the top matching documents
        top_indices = np.argpartition(scores, -num_results)[-num_results:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        # return the top documents
        top_docs = [self.docs[i] for i in top_indices]
        return top_docs
