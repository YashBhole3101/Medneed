from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)

def predict(values):
    if len(values) == 8:
        model = pickle.load(open('models/model_dia.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 13:
        model = pickle.load(open('models/heart.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 15:
        model = pickle.load(open('models/rfcModel.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 10:
        model = pickle.load(open('models/liver.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

@app.route("/")
def home():
    return render_template('home_page.html')

@app.route("/diabetes_page", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes_page.html')

@app.route("/heart_page", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart_page.html')

@app.route("/liver_page", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver_page.html')

@app.route("/lung_page", methods=['GET', 'POST'])
def lungPage():
    return render_template('lung_page.html')

@app.route("/prediction_page", methods = ['POST', 'GET'])
def predictPage():
    if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            # print(to_predict_dict,to_predict_list)
            pred = predict(to_predict_list)
            # print(pred)
    return render_template('prediction_page.html', pred = pred)

if __name__ == '__main__':
	app.run(debug = True)