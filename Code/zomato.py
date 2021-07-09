import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
model = pickle.load(open('zomato.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index2.html')


@app.route('/predict',methods=['POST'])
def predict():
    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 1)
    return render_template('index2.html', prediction_text='Your Rating is: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)