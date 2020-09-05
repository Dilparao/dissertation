# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 17:04:59 2020

@author: Dilba
"""


from flask import Flask, request, render_template
import os
import pickle

print(os.getcwd())
path = os.getcwd()

app = Flask(__name__)

with open('dt.pkl', 'rb') as f:
    dt_model = pickle.load(f)

with open('lr.pkl', 'rb') as f:
    logistic_model = pickle.load(f)

with open('svm_linear.pkl', 'rb') as f:
    svm_linear_model = pickle.load(f)

with open('svm_poly.pkl', 'rb') as f:
     svm_poly_model = pickle.load(f)
    
with open('svm_rbf.pkl', 'rb') as f:
    svm_rbf_model = pickle.load(f)

with open('rf.pkl', 'rb') as f:
     rf_model = pickle.load(f)


def get_predictions(cholesterol, gluc, final_age, final_ap_hi, final_ap_lo, final_bmi, req_model):
    mylist = [cholesterol, gluc, final_age, final_ap_hi, final_ap_lo, final_bmi]
    mylist = [float(i) for i in mylist]
    vals = [mylist]

    if req_model == 'Logistic':
        return logistic_model.predict(vals)[0]
    elif req_model == 'DecisionTree':
         return dt_model.predict(vals)[0]
    elif req_model == 'SvmLinear':
         return svm_linear_model.predict(vals)[0]    
    elif req_model == 'SvmPoly':
        return svm_poly_model.predict(vals)[0]
    elif req_model == 'SvmRbf':
        return svm_rbf_model.predict(vals)[0]        
    elif req_model == 'RandomForest':
         return rf_model.predict(vals)[0]    
    else:
        return "Cannot Predict"




@app.route('/')
def my_form():
    return render_template('home.html')

@app.route('/', methods=['POST', 'GET'])
def my_form_post():
    cholesterol=request.form['cholesterol']
    gluc = request.form['gluc']
    final_age = request.form['final_age']    
    final_ap_hi = request.form['final_ap_hi']
    final_ap_lo = request.form['final_ap_lo']
    final_bmi = request.form['final_bmi']
    req_model = request.form['req_model']
    print(cholesterol, gluc, final_age, final_ap_hi, final_ap_lo, final_bmi, req_model)
    target = get_predictions(cholesterol, gluc, final_age, final_ap_hi, final_ap_lo, final_bmi, req_model)
    

    if target==1:
        disease_present = 'Patient is likely to have heart disease'
    else:
        disease_present = 'Patient is unlikely to have heart disease'

    return render_template('home.html', target = target, disease_present = disease_present)


if __name__ == "__main__":
    app.run(debug=True)