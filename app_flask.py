from PIL import Image
import numpy as np
from flask import Flask, request, jsonify, render_template,redirect, url_for , flash
import pickle
import json
import pandas as pd
from functions_helper import *

app = Flask(__name__)

df_cid_features = pd.read_csv('TestData/TestData_merged_all.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    c_id = request.form.get('c_id')
    print("\n\n CID ")
    print(c_id)

    cid_data = df_cid_features.query("CustomerID==@c_id")
    prediction_df = get_preds(cid_data)
    output = prediction_df['predictions'].iloc[0]

    return render_template('index.html', prediction_text='Prediction {}'.format(output))


@app.route('/upload_files',methods=['POST'])
def upload_files():
   if request.method == 'POST':
    # check if the post request has the file part
    if 'demo_file' not in request.files:
     flash('No file part')
     return redirect(request.url)

    demo_file = request.files['demo_file']
    demo_file = pd.read_csv(demo_file)

    policy_file = request.files['policy_file']
    policy_file = pd.read_csv(policy_file)

    claim_file = request.files['claim_file']
    claim_file = pd.read_csv(claim_file)

    vehicle_file = request.files['vehicle_file']
    vehicle_file = pd.read_csv(vehicle_file)

    demo_output_shape = str(demo_file.shape[0])+'-'+str(demo_file.shape[1])
    policy_output_shape = str(policy_file.shape[0])+'-'+str(policy_file.shape[1])
    claim_output_shape = str(claim_file.shape[0]) + '-' + str(claim_file.shape[1])
    vehicle_output_shape = str(vehicle_file.shape[0]) + '-' + str(vehicle_file.shape[1])

    PP_all_test_data_custom(demo_file,claim_file,policy_file,vehicle_file)

    df_merged = pd.read_csv("TestData/TestData_merged_all_custom.csv")
    prediction_df = get_preds(df_merged)

    prediction_df.to_csv("TestData/TestData_merged_all_custom_with_preds.csv")

    return render_template('index.html',
                           demo_input_file_shape='shape {}'.format(demo_output_shape),
                           policy_input_file_shape='shape {}'.format(policy_output_shape),
                           claim_input_file_shape='shape {}'.format(claim_output_shape),
                           vehicle_input_file_shape='shape {}'.format(vehicle_output_shape),
                           )


if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host='0.0.0.0',debug=True)