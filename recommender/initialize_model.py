import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def initialize(small_version=False):
    """
    Initializes the paper-titles dataframe and Tfidf-Vectorizers used for the recommend module.

    Can run a small version for debugging which loads faster into memory.    

    Parameters
    ----------
    small_version : boolean
        Determines if the debug versions of the model is loaded (Default value = False).

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

    if small_version==True:
        fp = 'arxiv_sample/sample_'
        print('Small debug dataset in use.')
    else:
        fp = 'arxiv_data/tfidf_'
        print('Full dataset in use.')

    # Load data
    math_papers = pd.read_pickle('recommender/' + fp + 'math_arxiv_titles.pkl')

    # Load model data
    tfidf_title_vectorizer = joblib.load('recommender/' + fp + 'title_vectorizer.pkl')
    tfidf_abstract_vectorizer = joblib.load('recommender/' + fp + 'abstract_vectorizer.pkl')

    tfidf_title_matrix = joblib.load('recommender/' + fp + 'title_matrix.pkl')
    tfidf_abstract_matrix = joblib.load('recommender/' + fp + 'abstract_matrix.pkl')

    return math_papers, tfidf_title_vectorizer, tfidf_abstract_vectorizer, tfidf_title_matrix, tfidf_abstract_matrix


math_papers, tfidf_title_vectorizer, tfidf_abstract_vectorizer, tfidf_title_matrix, tfidf_abstract_matrix = initialize(small_version=True)
