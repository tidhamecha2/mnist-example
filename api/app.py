#run with "flask run" command
from flask import Flask
from flask import request
from mnist_example.utils import load
import numpy as np
app = Flask(__name__)

best_model_path = './models/best_model/model.joblib'
clf = load(best_model_path)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route('/predict', methods=['POST'])
def predict():    
    input_json = request.json
    image = input_json['image']
    image = np.array(image).reshape(1, -1)
    predicted = clf.predict(image)
    return str(predicted[0])


