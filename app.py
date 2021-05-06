import flask
import os
import pickle
import pandas as pd
from skimage import io
from skimage import transform

app = flask.Flask(__name__,template_folder='templates')

path_to_text_classifier = 'models/text_classifier.pkl'
path_to_vectorizer = 'models/vectorizer.pkl'

with open(path_to_text_classifier, 'rb') as f:
    model = pickle.load(f)

with open(path_to_vectorizer,'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/', methods=['GET','POST'])
def main():
   if flask.request.method == 'GET':
       return(flask.render_template('index.html'))
    if flask.request.method == 'POST':
        user_input_text = flask.request.form['input'] 
        X = vectorizer.transform([user_input_text])
        predictions = model.predict(X)
        prediction = predictions[0]
        predicted_probas = model.predict_proba(X)
        predicted_proba = predicted_probas[0]
        precent_democrat = predicted_proba[0]
        precent_republican = predicted_proba[1]
        return flask.render_temple('index.html', 
            input_text=user_input_text,
            result = prediction, 
            precent_democrat=precent_democrat,
            precent_republican=precent_republican)

if __name__ =='__main__':
    app.run(debug=True)