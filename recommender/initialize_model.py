import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def initialize():
    """
    Initializes the paper-titles dataframe and Tfidf-Vectorizers used for the recommend module.

    Parameters
    ----------


    Returns
    -------
    math_papers: DataFrame
        Contains title and id data on the math papers used from the arXiv. Has columns ['id', 'authors', 'title', 'math_subjects', 'update_date',
       'creation_date']

    tfidf_title_vectorizer: TfidfVectorizer
        A tfidf-vectorizer for math paper titles.

    tfidf_abstract_vectorizer: TfidfVectorizer
        A tfidf-vectorizer for math paper abstracts.

    tfidf_title_matrix: csr_matrix
        A matrix of the tfidf-vectorized titles for each math paper used from the arXiv.

    tfidf_abstract_matrix: csr_matrix
        A matrix of the tfidf-vectorized abstract for each math paper used from the arXiv.
    """


    # Load data
    math_papers = pd.read_pickle('recommender/arxiv_data/tfidf_math_arxiv_titles.pkl')

    # Load model data
    tfidf_title_vectorizer = joblib.load('recommender/arxiv_data/tfidf_title_vectorizer.pkl')
    tfidf_abstract_vectorizer = joblib.load('recommender/arxiv_data/tfidf_abstract_vectorizer.pkl')

    tfidf_title_matrix = joblib.load('recommender/arxiv_data/tfidf_title_matrix.pkl')
    tfidf_abstract_matrix = joblib.load('recommender/arxiv_data/tfidf_abstract_matrix.pkl')

    return math_papers, tfidf_title_vectorizer, tfidf_abstract_vectorizer, tfidf_title_matrix, tfidf_abstract_matrix


math_papers, tfidf_title_vectorizer, tfidf_abstract_vectorizer, tfidf_title_matrix, tfidf_abstract_matrix = initialize()
