import flask
import os
import pickle
import pandas as pd
from skimage import io
from skimage import transform

app = flask.Flask(__name__,template_folder='templates')

path_to_text_classifier = ''

with open(path_to_text_classifier, 'rb') as f:
    model = pickle.load(f)


@app.route('/', methods=['GET','POST'])
def main():
    