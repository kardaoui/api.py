import pickle

import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/score', methods=["GET", "POST"])
def credit():
    data = pd.DataFrame([request.get_json()])
    score = get_score(data)
    return jsonify(score)

@app.before_first_request
def load_model():
    global clf
    """loading the trained model"""
    with open('model/model_shap.pkl', 'rb') as mdl:
        clf = pickle.load(mdl)


def get_score(data_client):
    score = clf.predict_proba(data_client)
    keys = (0, 1)
    return dict(zip(keys, score[0]))

# lancement de l'application
if __name__ == "__main__":
    app.run(debug=True)
