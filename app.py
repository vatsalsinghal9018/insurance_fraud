# fetaures_to_model=['InsuredAge',
#  'CapitalGains',
#  'CapitalLoss',
#  'InsuredGender_FEMALE',
#  'InsuredGender_MALE',
#  'InsuredEducationLevel_Associate',
#  'InsuredEducationLevel_College',
#  'InsuredEducationLevel_High School',
#  'InsuredEducationLevel_JD',
#  'InsuredEducationLevel_MD',
#  'InsuredEducationLevel_Masters',
#  'InsuredEducationLevel_PhD',
#  'InsuredOccupation_adm-clerical',
#  'InsuredOccupation_armed-forces',
#  'InsuredOccupation_craft-repair',
#  'InsuredOccupation_exec-managerial',
#  'InsuredOccupation_farming-fishing',
#  'InsuredOccupation_handlers-cleaners',
#  'InsuredOccupation_machine-op-inspct',
#  'InsuredOccupation_other-service',
#  'InsuredOccupation_priv-house-serv',
#  'InsuredOccupation_prof-specialty',
#  'InsuredOccupation_protective-serv',
#  'InsuredOccupation_sales',
#  'InsuredOccupation_tech-support',
#  'InsuredOccupation_transport-moving',
#  'InsuredHobbies_base-jumping',
#  'InsuredHobbies_basketball',
#  'InsuredHobbies_board-games',
#  'InsuredHobbies_bungie-jumping',
#  'InsuredHobbies_camping',
#  'InsuredHobbies_chess',
#  'InsuredHobbies_cross-fit',
#  'InsuredHobbies_dancing',
#  'InsuredHobbies_exercise',
#  'InsuredHobbies_golf',
#  'InsuredHobbies_hiking',
#  'InsuredHobbies_kayaking',
#  'InsuredHobbies_movies',
#  'InsuredHobbies_paintball',
#  'InsuredHobbies_polo',
#  'InsuredHobbies_reading',
#  'InsuredHobbies_skydiving',
#  'InsuredHobbies_sleeping',
#  'InsuredHobbies_video-games',
#  'InsuredHobbies_yachting',
#  'Country_India',
#  'CustomerLoyaltyPeriod',
#  'Policy_Deductible',
#  'PolicyAnnualPremium',
#  'UmbrellaLimit',
#  'InsurancePolicyState_State1',
#  'InsurancePolicyState_State2',
#  'InsurancePolicyState_State3',
#  'Policy_CombinedSingleLimit_100/1000',
#  'Policy_CombinedSingleLimit_100/300',
#  'Policy_CombinedSingleLimit_100/500',
#  'Policy_CombinedSingleLimit_250/1000',
#  'Policy_CombinedSingleLimit_250/300',
#  'Policy_CombinedSingleLimit_250/500',
#  'Policy_CombinedSingleLimit_500/1000',
#  'Policy_CombinedSingleLimit_500/300',
#  'Policy_CombinedSingleLimit_500/500',
#  'InsuredRelationship_husband',
#  'InsuredRelationship_not-in-family',
#  'InsuredRelationship_other-relative',
#  'InsuredRelationship_own-child',
#  'InsuredRelationship_unmarried',
#  'InsuredRelationship_wife',
#  'IncidentTime',
#  'NumberOfVehicles',
#  'BodilyInjuries',
#  'AmountOfTotalClaim',
#  'AmountOfInjuryClaim',
#  'AmountOfPropertyClaim',
#  'AmountOfVehicleDamage',
#  'TypeOfIncident_Multi-vehicle Collision',
#  'TypeOfIncident_Parked Car',
#  'TypeOfIncident_Single Vehicle Collision',
#  'TypeOfIncident_Vehicle Theft',
#  'TypeOfCollission_Front Collision',
#  'TypeOfCollission_Rear Collision',
#  'TypeOfCollission_Side Collision',
#  'SeverityOfIncident_Major Damage',
#  'SeverityOfIncident_Minor Damage',
#  'SeverityOfIncident_Total Loss',
#  'SeverityOfIncident_Trivial Damage',
#  'AuthoritiesContacted_Ambulance',
#  'AuthoritiesContacted_Fire',
#  'AuthoritiesContacted_None',
#  'AuthoritiesContacted_Other',
#  'AuthoritiesContacted_Police',
#  'IncidentState_State3',
#  'IncidentState_State4',
#  'IncidentState_State5',
#  'IncidentState_State6',
#  'IncidentState_State7',
#  'IncidentState_State8',
#  'IncidentState_State9',
#  'IncidentCity_City1',
#  'IncidentCity_City2',
#  'IncidentCity_City3',
#  'IncidentCity_City4',
#  'IncidentCity_City5',
#  'IncidentCity_City6',
#  'IncidentCity_City7',
#  'PropertyDamage_NO',
#  'PropertyDamage_YES',
#  'Witnesses_0',
#  'Witnesses_1',
#  'Witnesses_2',
#  'Witnesses_3',
#  'Witnesses_MISSINGVALUE',
#  'PoliceReport_NO',
#  'PoliceReport_YES',
#  'VehicleAttributeDetails_VehicleMake_Accura',
#  'VehicleAttributeDetails_VehicleMake_Audi',
#  'VehicleAttributeDetails_VehicleMake_BMW',
#  'VehicleAttributeDetails_VehicleMake_Chevrolet',
#  'VehicleAttributeDetails_VehicleMake_Dodge',
#  'VehicleAttributeDetails_VehicleMake_Ford',
#  'VehicleAttributeDetails_VehicleMake_Honda',
#  'VehicleAttributeDetails_VehicleMake_Jeep',
#  'VehicleAttributeDetails_VehicleMake_Mercedes',
#  'VehicleAttributeDetails_VehicleMake_Nissan',
#  'VehicleAttributeDetails_VehicleMake_Saab',
#  'VehicleAttributeDetails_VehicleMake_Suburu',
#  'VehicleAttributeDetails_VehicleMake_Toyota',
#  'VehicleAttributeDetails_VehicleMake_Volkswagen',
#  'VehicleAttributeDetails_VehicleModel_3 Series',
#  'VehicleAttributeDetails_VehicleModel_92x',
#  'VehicleAttributeDetails_VehicleModel_93',
#  'VehicleAttributeDetails_VehicleModel_95',
#  'VehicleAttributeDetails_VehicleModel_A3',
#  'VehicleAttributeDetails_VehicleModel_A5',
#  'VehicleAttributeDetails_VehicleModel_Accord',
#  'VehicleAttributeDetails_VehicleModel_C300',
#  'VehicleAttributeDetails_VehicleModel_CRV',
#  'VehicleAttributeDetails_VehicleModel_Camry',
#  'VehicleAttributeDetails_VehicleModel_Civic',
#  'VehicleAttributeDetails_VehicleModel_Corolla',
#  'VehicleAttributeDetails_VehicleModel_E400',
#  'VehicleAttributeDetails_VehicleModel_Escape',
#  'VehicleAttributeDetails_VehicleModel_F150',
#  'VehicleAttributeDetails_VehicleModel_Forrestor',
#  'VehicleAttributeDetails_VehicleModel_Fusion',
#  'VehicleAttributeDetails_VehicleModel_Grand Cherokee',
#  'VehicleAttributeDetails_VehicleModel_Highlander',
#  'VehicleAttributeDetails_VehicleModel_Impreza',
#  'VehicleAttributeDetails_VehicleModel_Jetta',
#  'VehicleAttributeDetails_VehicleModel_Legacy',
#  'VehicleAttributeDetails_VehicleModel_M5',
#  'VehicleAttributeDetails_VehicleModel_MDX',
#  'VehicleAttributeDetails_VehicleModel_ML350',
#  'VehicleAttributeDetails_VehicleModel_Malibu',
#  'VehicleAttributeDetails_VehicleModel_Maxima',
#  'VehicleAttributeDetails_VehicleModel_Neon',
#  'VehicleAttributeDetails_VehicleModel_Passat',
#  'VehicleAttributeDetails_VehicleModel_Pathfinder',
#  'VehicleAttributeDetails_VehicleModel_RAM',
#  'VehicleAttributeDetails_VehicleModel_RSX',
#  'VehicleAttributeDetails_VehicleModel_Silverado',
#  'VehicleAttributeDetails_VehicleModel_TL',
#  'VehicleAttributeDetails_VehicleModel_Tahoe',
#  'VehicleAttributeDetails_VehicleModel_Ultima',
#  'VehicleAttributeDetails_VehicleModel_Wrangler',
#  'VehicleAttributeDetails_VehicleModel_X5',
#  'VehicleAttributeDetails_VehicleModel_X6',
#  'no_days_incident_vehicleYOM',
#  'no_days_incident_PolicyCoverage']
# # import streamlit as st
# from PIL import Image
# import numpy as np
# from flask import Flask, request, jsonify, render_template
# import pickle
# import json
# import pandas as pd
#
# app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))
# df_cid_features=pd.read_csv('output.csv')
#
# @app.route('/predict',methods=['POST'])
# def predict(c_id):
#
#     cid_data = df_cid_features.query("CustomerID==@c_id")
#     prediction = model.predict(cid_data[fetaures_to_model])
#     # return
#     return prediction
#
# def main():
#     st.title("Insurance Fraud Prediction")
#     html_temp = """
#     <div style="background-color:tomato;padding:10px">
#     <h2 style="color:white;text-align:center;">Streamlit IRIS Predictor </h2>
#     </div>
#     """
#     st.markdown(html_temp,unsafe_allow_html=True)
#     c_id = st.text_input("Customer ID","Type Here")
#
#     result=""
#     if st.button("Predict"):
#         result=predict(c_id)
#     st.success('The output is {}'.format(result))
#     if st.button("About"):
#         st.text("Lets LEarn")
#         st.text("Built with Streamlit")
# if __name__=='__main__':
#     main()
#
#
#
# @app.route('/')
# def welcome():
#     return "Index Page"
