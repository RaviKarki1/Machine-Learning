# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 09:55:04 2020

@author: ravik
"""


from flask import Flask, render_template, request
#import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

app = Flask(__name__) 
model = pickle.load(open('car_prediction.pkl', 'rb'))


@app.route("/", methods=['GET'])
def home():
    return render_template("Index.html")


standard_to = StandardScaler()

@app.route('/predict', methods=['POST'])
def predict():
    Fuel_Type_Diesel = 0
    if request == "POST":
        
        Year = int(request.form['year'])
        
        present_price = float(request.form("present_price"))
        
        Kms_Driven = float(request.form(['Kms_Driven']))
        
        Owner = int(request.form(['Owner']))
        
        Fuel_Type_Petrol = request.form(['Fuel_Type_Petrol'])
        if(Fuel_Type_Petrol == "Petrol"):
            Fuel_Type_Petrol = 1
            Fuel_Type_Diesel = 0
        else:
            Fuel_Type_Petrol = 0
            Fuel_Type_Diesel = 1
            
        Seller_Type_Individual = request.form(['Seller_Type_Individual'])
        if(Seller_Type_Individual == "Individaul"):
            Seller_Type_Individual = 1
        else: 
            Seller_Type_Individual = 0
            
        Transmission_Mannual = request.form(['Transmission_Mannual'])
        if(Transmission_Mannual == "Mannual"):
            Transmission_Mannual = 1
        else:
            Transmission_Mannual = 0
            
        prediction=model.predict([[Present_Price,
                                   Kms_Driven2,
                                   Owner,
                                   Year,
                                   Fuel_Type_Diesel,
                                   Fuel_Type_Petrol,
                                   Seller_Type_Individual,
                                   Transmission_Mannual
                                   ]])
                
        output=round(prediction[0],2)
        
        if output<0:
            return render_template('index.html',prediction_texts="Sorry you cannot sell this car")
        else:
            return render_template('index.html',prediction_text="You Can Sell The Car at {}".format(output))
        
    else:
        return render_template('index.html')
        
        











if __name__ == "__main__":
    app.run(debug = True)
