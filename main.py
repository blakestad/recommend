"""
The main web api for the recommender package.

Takes POST requests with the data of a paper title and abstract, returns a collection of titles and arxiv id's of related papers.
"""

import pandas as pd
import joblib
import os
from flask import Flask, request, jsonify

from recommender import recommend_similiar_paper

app = Flask(__name__)

@app.route('/', methods=['POST'])
def recommend():
    data = request.json
    title=data.get('input_title', None)
    abstract=data.get('input_abstract', None)
    results = recommend_similiar_paper(input_title=title, input_abstract=abstract)
    results['math_subjects'] = results['math_subjects'].map(lambda x: list(x))
    results = results.reset_index()
    results_dict = results.to_dict()
    results_json = jsonify(results_dict)
    return jsonify(results.to_dict())


if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=8080)
    app.run(debug=True)