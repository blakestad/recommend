from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

from recommender.initialize_model import math_papers, tfidf_title_vectorizer, tfidf_abstract_vectorizer, tfidf_title_matrix, tfidf_abstract_matrix

def recommend_similiar_paper(input_title=None, input_abstract=None, n=10, title_weight=0.5, abstract_weight=1):
    """

    Parameters
    ----------
    input_title : string
        Title of input paper (Default value = None)
    input_abstract : string
        Abstract of the input paper (Default value = None)
    n : integer
        Number of papers to recommend (Default value = 10)
    title_weight : numeric
        Relative weighting of the title (Default value = 0.5)
    abstract_weight : numeric
        relative weighting of the abstract (Default value = 1)

    Returns
    -------
    top_papers: DataFrame
        The papers recommended from the model.
    """
    if (input_title == None) and (input_abstract == None):
        print('No inputs')
        return None
    elif(input_title != None):
        title_scores = linear_kernel(tfidf_title_vectorizer.transform([input_title]),
                                     tfidf_title_matrix)
        total_scores = title_scores[0]
    
    elif(input_abstract != None):
        abstract_scores = linear_kernel(tfidf_abstract_vectorizer.transform([input_abstract]),
                                       tfidf_abstract_matrix)
        total_scores = abstract_scores[0]
        
    else:
        title_scores = linear_kernel(tfidf_title_vectorizer.transform([input_title]),
                                     tfidf_title_matrix)
        abstract_scores = linear_kernel(tfidf_abstract_vectorizer.transform([input_abstract]),
                                       tfidf_abstract_matrix)
        total_scores = title_weight*title_scores[0] + abstract_weight*abstract_scores[0]
        
    top_indices = total_scores.argsort()[(-1)*n:][::-1]
    top_papers = math_papers.iloc[top_indices]

    top_papers=top_papers.reset_index(drop=True)
    
    return top_papers





