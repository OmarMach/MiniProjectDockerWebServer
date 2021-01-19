from flask import Flask, render_template, request
from joblib import load

import librosa
import os
import pandas as pd

from audioFileConversion import get_features
from werkzeug.utils import secure_filename


app = Flask('MiniProjetDocker')


@app.route('/')
def index_page():
    return render_template('predictorform.html')


@app.route('/results', methods=['POST'])
def results():
    if request.method == 'POST':
        if not request.files['file']:
            return render_template('resultsform.html', error='Please provide a valid file.')

        file = request.files['file']

        model = load(filename='./models/svc.joblib')

        signal, sr = librosa.load(file)
        features = get_features(signal, sr)

        result = model.predict(features)

        print('The predicted genre is using SVM is : %s' % result[0])
        print('The predicted genre is using kNN is : %s' % result[0])

        return render_template('resultsform.html',  predicted_genre=result[0])


app.run("localhost", "9999", debug=True)
