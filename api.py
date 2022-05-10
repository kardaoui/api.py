from flask import Flask, jsonify, request

import pandas as pd
import numpy as np

import pickle

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def credit():
    data = pd.DataFrame([request.get_json()])
    score = get_score(data)
    return jsonify(score)

def load_model():
    """loading the trained model"""
    with open('model/model_test.pkl', 'rb') as mdl:
        clf = pickle.load(mdl)
    return clf

def get_score(data_client):
    clf = load_model()
    score = clf.predict_proba(data_client)
    keys = (0, 1)
    return dict(zip(keys, score[0]))

# lancement de l'application
if __name__ == "__main__":
    app.run(debug=True)
