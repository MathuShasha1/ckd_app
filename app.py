# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Decision Tree CLassifier model
# filename = '/ckd_model.pkl'



app = Flask(__name__)
classifier = pickle.load(open("ckd_model.pkl", 'rb'))



@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        bp = float(request.form['bp'])
        sg = float(request.form['sg'])
        al = float(request.form['al'])
        bgr = float(request.form['bgr'])
        bu = float(request.form['bu'])
        sc = float(request.form['sc'])
        sod = float(request.form['sod'])
        pot = float(request.form['pot'])
        hemo = float(request.form['hemo'])
        pcv = float(request.form['pcv'])
        wbcc = float(request.form['wbcc'])
        rbcc = float(request.form['rbcc'])
        rbc_normal = int(request.form['rbc_normal'])
        pc_normal  = int(request.form['pc_normal'])
        htn_yes = int(request.form['htn_yes'])
        dm_yes = int(request.form['dm_yes'])
		
        data = np.array([[age, bp, sg, al, bgr, bu, sc, sod, pot, hemo, pcv, wbcc, rbcc, rbc_normal, pc_normal, htn_yes, dm_yes]])
        my_prediction = classifier.predict(data)
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)