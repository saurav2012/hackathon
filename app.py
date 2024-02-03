from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('new_prediction.pkl', 'rb'))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def home(): 
    data1 = int(request.form['Year'])
    data2 = request.form['Stage']
    data3 = request.form['Stadium']
    data4 = request.form['City']
    data5 = request.form['Home Team Name']
    data6 = request.form['Away Team Name']

    new_data = pd.DataFrame({'Year': [data1],
                         'Stage': [data2],
                         'Stadium': [data3],
                         'City': [data4],
                         'Home Team Name': [data5],
                         'Away Team Name': [data6]})
    
    # Make predictions on new data
    new_prediction = model.predict(new_data)

    return jsonify({"result": new_prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)