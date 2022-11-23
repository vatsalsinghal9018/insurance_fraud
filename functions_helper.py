import json
import pickle
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,auc, accuracy_score,f1_score,recall_score,precision_score, confusion_matrix, mean_squared_error,roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import xgboost as xgb


def process_vehicle_train_data(df):
    df = df.dropna(subset=['CustomerID'])

    if ('VehicleAttributeDetails' in df.columns):
        df['VehicleAttributeDetails'] = np.where(df['VehicleAttributeDetails'] == '???',
                                                 np.nan, df['VehicleAttributeDetails'])

    df = df.pivot(index=['CustomerID'], columns=['VehicleAttribute'], values=['VehicleAttributeDetails'])
    df.columns = ['_'.join(x).strip() for x in df.columns]
    df = df.reset_index()
    df = df.drop(['VehicleAttributeDetails_VehicleID'], axis=1)
    df['VehicleAttributeDetails_VehicleYOM'] = pd.to_datetime(df['VehicleAttributeDetails_VehicleYOM']
                                                              , format='%Y')

    cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    print("cat_cols. :", cat_cols)
    num_cols = df.select_dtypes(exclude=['datetime', 'object', 'string', 'category']).columns.tolist()
    print("\nnum_cols. :", num_cols)
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    print("\ndatetime_cols.  :", datetime_cols)

    imp_mean = SimpleImputer(strategy='mean')
    imp_mode = SimpleImputer(strategy='most_frequent')

    df_num = df[num_cols]
    df_cat = df[cat_cols]
    df_datetime = df[datetime_cols]

    if (len(num_cols)) > 0:
        df_num = pd.DataFrame(imp_mean.fit_transform(df_num), columns=df_num.columns)
    if (len(cat_cols)) > 0:
        df_cat = pd.DataFrame(imp_mode.fit_transform(df_cat), columns=df_cat.columns)

    df_pp = pd.concat([df_cat, df_num, df_datetime], axis=1)

    oh_enc = OneHotEncoder(handle_unknown='ignore')
    cat_cols_onc = [i for i in cat_cols if i not in ('CustomerID', 'InsuredZipCode')]
    if (len(cat_cols_onc) > 0):
        print("\n cat_cols_onc. ::", cat_cols_onc)
        df_enc = pd.DataFrame(oh_enc.fit_transform(df_pp[cat_cols_onc]).toarray(),
                              columns=oh_enc.get_feature_names(cat_cols_onc))
        df_pp = df_pp.drop(cat_cols_onc, axis=1)
        df_pp = pd.concat([df_pp, df_enc], axis=1)

    dictionary_metadata = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "datetime_cols": datetime_cols,
        "cat_cols_onc": cat_cols_onc
    }

    json_object = json.dumps(dictionary_metadata, indent=4)
    with open("data_pre_process_models/vehicle_metadata.json", "w") as outfile:
        outfile.write(json_object)

    pickle.dump(imp_mean, open('data_pre_process_models/vehicle_imp_mean.pkl', 'wb'))
    pickle.dump(imp_mode, open('data_pre_process_models/vehicle_imp_mode.pkl', 'wb'))
    pickle.dump(oh_enc, open('data_pre_process_models/vehicle_oh_enc.pkl', 'wb'))

    return df_pp


def process_vehicle_test_data(df):
    df = df.dropna(subset=['CustomerID'])

    if ('VehicleAttributeDetails' in df.columns):
        df['VehicleAttributeDetails'] = np.where(df['VehicleAttributeDetails'] == '???',
                                                 np.nan, df['VehicleAttributeDetails'])

    df = df.pivot(index=['CustomerID'], columns=['VehicleAttribute'], values=['VehicleAttributeDetails'])
    df.columns = ['_'.join(x).strip() for x in df.columns]
    df = df.reset_index()
    if('VehicleAttributeDetails_VehicleID' in df.columns):
        df = df.drop(['VehicleAttributeDetails_VehicleID'], axis=1)
    if('VehicleAttributeDetails_VehicleYOM' in df.columns):
        df['VehicleAttributeDetails_VehicleYOM'] = pd.to_datetime(df['VehicleAttributeDetails_VehicleYOM']
                                                              , format='%Y')

    with open('data_pre_process_models/vehicle_metadata.json', 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
        dictionary_metadata = dict(json_object)
    num_cols = dictionary_metadata['num_cols']
    cat_cols = dictionary_metadata['cat_cols']
    datetime_cols = dictionary_metadata['datetime_cols']
    cat_cols_onc = dictionary_metadata['cat_cols_onc']
    print("num_cols.  :", num_cols)
    print("cat_cols.  :", cat_cols)
    print("datetime_cols.  :", datetime_cols)
    print("cat_cols_onc.  :", cat_cols_onc)

    for i in num_cols+cat_cols+datetime_cols:
        if i not in df.columns:
            print("columns {col_i} not in df test , adding this columns".format(col_i=str(i)))
            df[i] = np.nan

    imp_mean = pickle.load(open('data_pre_process_models/vehicle_imp_mean.pkl', 'rb'))
    imp_mode = pickle.load(open('data_pre_process_models/vehicle_imp_mode.pkl', 'rb'))

    df_num = df[num_cols]
    df_cat = df[cat_cols]
    df_datetime = df[datetime_cols]

    if (len(num_cols)) > 0:
        df_num = pd.DataFrame(imp_mean.transform(df_num), columns=df_num.columns)
    if (len(cat_cols)) > 0:
        df_cat = pd.DataFrame(imp_mode.transform(df_cat), columns=df_cat.columns)

    df_pp = pd.concat([df_cat, df_num, df_datetime], axis=1)

    oh_enc = pickle.load(open('data_pre_process_models/vehicle_oh_enc.pkl', 'rb'))

    if (len(cat_cols_onc) > 0):
        print("\n cat_cols_onc. ::", cat_cols_onc)
        df_enc = pd.DataFrame(oh_enc.transform(df_pp[cat_cols_onc]).toarray(),
                              columns=oh_enc.get_feature_names(cat_cols_onc))
        df_pp = df_pp.drop(cat_cols_onc, axis=1)
        df_pp = pd.concat([df_pp, df_enc], axis=1)



    return df_pp


def process_policy_train_data(df):
    df = df.dropna(subset=['CustomerID'])

    # POLICY INFO
    if ('InsurancePolicyNumber' in df.columns):
        df = df.drop(['InsurancePolicyNumber'], axis=1)

    if ('PolicyAnnualPremium' in df.columns):
        df['PolicyAnnualPremium'] = np.where(df['PolicyAnnualPremium'] == -1,
                                             np.nan, df['PolicyAnnualPremium'])
    if ('DateOfPolicyCoverage' in df.columns):
        df['DateOfPolicyCoverage'] = pd.to_datetime(df['DateOfPolicyCoverage'], format='%Y-%m-%d')

    cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    print("cat_cols. :", cat_cols)
    num_cols = df.select_dtypes(exclude=['datetime', 'object', 'string', 'category']).columns.tolist()
    print("\nnum_cols. :", num_cols)
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    print("\ndatetime_cols.  :", datetime_cols)

    imp_mean = SimpleImputer(strategy='mean')
    imp_mode = SimpleImputer(strategy='most_frequent')

    df_num = df[num_cols]
    df_cat = df[cat_cols]
    df_datetime = df[datetime_cols]

    if (len(num_cols)) > 0:
        df_num = pd.DataFrame(imp_mean.fit_transform(df_num), columns=df_num.columns)
    if (len(cat_cols)) > 0:
        df_cat = pd.DataFrame(imp_mode.fit_transform(df_cat), columns=df_cat.columns)

    df_pp = pd.concat([df_cat, df_num, df_datetime], axis=1)

    oh_enc = OneHotEncoder(handle_unknown='ignore')
    cat_cols_onc = [i for i in cat_cols if i not in ('CustomerID', 'InsuredZipCode')]
    if (len(cat_cols_onc) > 0):
        print("\n cat_cols_onc. ::", cat_cols_onc)
        df_enc = pd.DataFrame(oh_enc.fit_transform(df_pp[cat_cols_onc]).toarray(),
                              columns=oh_enc.get_feature_names(cat_cols_onc))
        df_pp = df_pp.drop(cat_cols_onc, axis=1)
        df_pp = pd.concat([df_pp, df_enc], axis=1)

    dictionary_metadata = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "datetime_cols": datetime_cols,
        "cat_cols_onc": cat_cols_onc
    }

    json_object = json.dumps(dictionary_metadata, indent=4)
    with open("data_pre_process_models/policy_metadata.json", "w") as outfile:
        outfile.write(json_object)

    pickle.dump(imp_mean, open('data_pre_process_models/policy_imp_mean.pkl', 'wb'))
    pickle.dump(imp_mode, open('data_pre_process_models/policy_imp_mode.pkl', 'wb'))
    pickle.dump(oh_enc, open('data_pre_process_models/policy_oh_enc.pkl', 'wb'))

    return df_pp


def process_policy_test_data(df):
    df = df.dropna(subset=['CustomerID'])
    # POLICY INFO
    if ('InsurancePolicyNumber' in df.columns):
        df = df.drop(['InsurancePolicyNumber'], axis=1)

    if ('PolicyAnnualPremium' in df.columns):
        df['PolicyAnnualPremium'] = np.where(df['PolicyAnnualPremium'] == -1,
                                             np.nan, df['PolicyAnnualPremium'])
    if ('DateOfPolicyCoverage' in df.columns):
        df['DateOfPolicyCoverage'] = pd.to_datetime(df['DateOfPolicyCoverage'], format='%Y-%m-%d')

    with open('data_pre_process_models/policy_metadata.json', 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
        dictionary_metadata = dict(json_object)
    num_cols = dictionary_metadata['num_cols']
    cat_cols = dictionary_metadata['cat_cols']
    datetime_cols = dictionary_metadata['datetime_cols']
    cat_cols_onc = dictionary_metadata['cat_cols_onc']
    print("num_cols.  :", num_cols)
    print("cat_cols.  :", cat_cols)
    print("datetime_cols.  :", datetime_cols)
    print("cat_cols_onc.  :", cat_cols_onc)

    imp_mean = pickle.load(open('data_pre_process_models/policy_imp_mean.pkl', 'rb'))
    imp_mode = pickle.load(open('data_pre_process_models/policy_imp_mode.pkl', 'rb'))

    df_num = df[num_cols]
    df_cat = df[cat_cols]
    df_datetime = df[datetime_cols]

    if (len(num_cols)) > 0:
        df_num = pd.DataFrame(imp_mean.transform(df_num), columns=df_num.columns)
    if (len(cat_cols)) > 0:
        df_cat = pd.DataFrame(imp_mode.transform(df_cat), columns=df_cat.columns)

    df_pp = pd.concat([df_cat, df_num, df_datetime], axis=1)

    oh_enc = pickle.load(open('data_pre_process_models/policy_oh_enc.pkl', 'rb'))

    if (len(cat_cols_onc) > 0):
        print("\n cat_cols_onc. ::", cat_cols_onc)
        df_enc = pd.DataFrame(oh_enc.transform(df_pp[cat_cols_onc]).toarray(),
                              columns=oh_enc.get_feature_names(cat_cols_onc))
        df_pp = df_pp.drop(cat_cols_onc, axis=1)
        df_pp = pd.concat([df_pp, df_enc], axis=1)

    return df_pp


def process_claim_train_data(df):
    df = df.dropna(subset=['CustomerID'])

    if ('TypeOfCollission' in df.columns):
        df['TypeOfCollission'] = np.where(df['TypeOfCollission'] == '?',
                                          np.nan, df['TypeOfCollission'])
    if ('IncidentTime' in df.columns):
        df['IncidentTime'] = np.where(df['IncidentTime'] == -5,
                                      np.nan, df['IncidentTime'])
    if ('PropertyDamage' in df.columns):
        df['PropertyDamage'] = np.where(df['PropertyDamage'] == '?',
                                        np.nan, df['PropertyDamage'])
    if ('PoliceReport' in df.columns):
        df['PoliceReport'] = np.where(df['PoliceReport'] == '?',
                                      np.nan, df['PoliceReport'])

    if ('Witnesses' in df.columns):
        df['Witnesses'] = np.where(df['Witnesses'] == 'MISSINGVALUE',
                                   np.nan, df['Witnesses'])
        # df['Witnesses']=df['Witnesses'].astype(float)

    if ('AmountOfTotalClaim' in df.columns):
        df['AmountOfTotalClaim'] = np.where(df['AmountOfTotalClaim'] == 'MISSEDDATA',
                                            np.nan, df['AmountOfTotalClaim'])
        df['AmountOfTotalClaim'] = df['AmountOfTotalClaim'].astype(float)

    if ('DateOfIncident' in df.columns):
        df['DateOfIncident'] = pd.to_datetime(df['DateOfIncident'], format='%Y-%m-%d')

    if ('IncidentAddress' in df.columns):
        df.drop(['IncidentAddress'], axis=1, inplace=True)

    for i in ['TypeOfCollission', 'PropertyDamage', 'PoliceReport']:
        if i in df.columns:
            df[i] = np.where(pd.isnull(df[i]), 'NA', df[i])

    cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    print("cat_cols. :", cat_cols)
    num_cols = df.select_dtypes(exclude=['datetime', 'object', 'string', 'category']).columns.tolist()
    print("\nnum_cols. :", num_cols)
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    print("\ndatetime_cols.  :", datetime_cols)

    imp_mean = SimpleImputer(strategy='mean')
    imp_mode = SimpleImputer(strategy='most_frequent')

    df_num = df[num_cols]
    df_cat = df[cat_cols]
    df_datetime = df[datetime_cols]

    if (len(num_cols)) > 0:
        df_num = pd.DataFrame(imp_mean.fit_transform(df_num), columns=df_num.columns)
    if (len(cat_cols)) > 0:
        df_cat = pd.DataFrame(imp_mode.fit_transform(df_cat), columns=df_cat.columns)

    df_pp = pd.concat([df_cat, df_num, df_datetime], axis=1)

    oh_enc = OneHotEncoder(handle_unknown='ignore')
    cat_cols_onc = [i for i in cat_cols if i not in ('CustomerID', 'InsuredZipCode')]
    if (len(cat_cols_onc) > 0):
        print("\n cat_cols_onc. ::", cat_cols_onc)
        df_enc = pd.DataFrame(oh_enc.fit_transform(df_pp[cat_cols_onc]).toarray(),
                              columns=oh_enc.get_feature_names(cat_cols_onc))
        df_pp = df_pp.drop(cat_cols_onc, axis=1)
        df_pp = pd.concat([df_pp, df_enc], axis=1)

    dictionary_metadata = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "datetime_cols": datetime_cols,
        "cat_cols_onc": cat_cols_onc
    }

    json_object = json.dumps(dictionary_metadata, indent=4)
    with open("data_pre_process_models/claim_metadata.json", "w") as outfile:
        outfile.write(json_object)

    pickle.dump(imp_mean, open('data_pre_process_models/claim_imp_mean.pkl', 'wb'))
    pickle.dump(imp_mode, open('data_pre_process_models/claim_imp_mode.pkl', 'wb'))
    pickle.dump(oh_enc, open('data_pre_process_models/claim_oh_enc.pkl', 'wb'))

    return df_pp


def process_claim_test_data(df):
    df = df.dropna(subset=['CustomerID'])

    if ('TypeOfCollission' in df.columns):
        df['TypeOfCollission'] = np.where(df['TypeOfCollission'] == '?',
                                          np.nan, df['TypeOfCollission'])
    if ('IncidentTime' in df.columns):
        df['IncidentTime'] = np.where(df['IncidentTime'] == -5,
                                      np.nan, df['IncidentTime'])
    if ('PropertyDamage' in df.columns):
        df['PropertyDamage'] = np.where(df['PropertyDamage'] == '?',
                                        np.nan, df['PropertyDamage'])
    if ('PoliceReport' in df.columns):
        df['PoliceReport'] = np.where(df['PoliceReport'] == '?',
                                      np.nan, df['PoliceReport'])

    if ('Witnesses' in df.columns):
        df['Witnesses'] = np.where(df['Witnesses'] == 'MISSINGVALUE',
                                   np.nan, df['Witnesses'])
        # df['Witnesses']=df['Witnesses'].astype(float)

    if ('AmountOfTotalClaim' in df.columns):
        df['AmountOfTotalClaim'] = np.where(df['AmountOfTotalClaim'] == 'MISSEDDATA',
                                            np.nan, df['AmountOfTotalClaim'])
        df['AmountOfTotalClaim'] = df['AmountOfTotalClaim'].astype(float)

    if ('DateOfIncident' in df.columns):
        df['DateOfIncident'] = pd.to_datetime(df['DateOfIncident'], format='%Y-%m-%d')

    if ('IncidentAddress' in df.columns):
        df.drop(['IncidentAddress'], axis=1, inplace=True)

    for i in ['TypeOfCollission', 'PropertyDamage', 'PoliceReport']:
        if i in df.columns:
            df[i] = np.where(pd.isnull(df[i]), 'NA', df[i])

    with open('data_pre_process_models/claim_metadata.json', 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
        dictionary_metadata = dict(json_object)
    num_cols = dictionary_metadata['num_cols']
    cat_cols = dictionary_metadata['cat_cols']
    datetime_cols = dictionary_metadata['datetime_cols']
    cat_cols_onc = dictionary_metadata['cat_cols_onc']
    print("num_cols.  :", num_cols)
    print("cat_cols.  :", cat_cols)
    print("datetime_cols.  :", datetime_cols)
    print("cat_cols_onc.  :", cat_cols_onc)

    imp_mean = pickle.load(open('data_pre_process_models/claim_imp_mean.pkl', 'rb'))
    imp_mode = pickle.load(open('data_pre_process_models/claim_imp_mode.pkl', 'rb'))

    df_num = df[num_cols]
    df_cat = df[cat_cols]
    df_datetime = df[datetime_cols]

    if (len(num_cols)) > 0:
        df_num = pd.DataFrame(imp_mean.transform(df_num), columns=df_num.columns)
    if (len(cat_cols)) > 0:
        df_cat = pd.DataFrame(imp_mode.transform(df_cat), columns=df_cat.columns)

    df_pp = pd.concat([df_cat, df_num, df_datetime], axis=1)

    oh_enc = pickle.load(open('data_pre_process_models/claim_oh_enc.pkl', 'rb'))

    if (len(cat_cols_onc) > 0):
        print("\n cat_cols_onc. ::", cat_cols_onc)
        df_enc = pd.DataFrame(oh_enc.transform(df_pp[cat_cols_onc]).toarray(),
                              columns=oh_enc.get_feature_names(cat_cols_onc))
        df_pp = df_pp.drop(cat_cols_onc, axis=1)
        df_pp = pd.concat([df_pp, df_enc], axis=1)

    return df_pp


def process_demo_train_data(df):
    df = df.dropna(subset=['CustomerID'])

    # DEMOGRAPHICS
    if ('InsuredGender' in df.columns):
        df['InsuredGender'] = np.where(df['InsuredGender'] == 'NA', np.nan, df['InsuredGender'])

    if ('InsuredZipCode' in df.columns):
        df = df.drop(['InsuredZipCode'], axis=1)
    if ('Country' in df.columns):
        df = df.drop(['Country'], axis=1)

    cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    print("cat_cols. :", cat_cols)
    num_cols = df.select_dtypes(exclude=['datetime', 'object', 'string', 'category']).columns.tolist()
    print("\nnum_cols. :", num_cols)
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    print("\ndatetime_cols.  :", datetime_cols)

    imp_mean = SimpleImputer(strategy='mean')
    imp_mode = SimpleImputer(strategy='most_frequent')

    df_num = df[num_cols]
    df_cat = df[cat_cols]
    df_datetime = df[datetime_cols]

    if (len(num_cols)) > 0:
        df_num = pd.DataFrame(imp_mean.fit_transform(df_num), columns=df_num.columns)
    if (len(cat_cols)) > 0:
        df_cat = pd.DataFrame(imp_mode.fit_transform(df_cat), columns=df_cat.columns)

    df_pp = pd.concat([df_cat, df_num, df_datetime], axis=1)

    oh_enc = OneHotEncoder(handle_unknown='ignore')
    cat_cols_onc = [i for i in cat_cols if i not in ('CustomerID', 'InsuredZipCode')]
    if (len(cat_cols_onc) > 0):
        print("\n cat_cols_onc. ::", cat_cols_onc)
        df_enc = pd.DataFrame(oh_enc.fit_transform(df_pp[cat_cols_onc]).toarray(),
                              columns=oh_enc.get_feature_names(cat_cols_onc))
        df_pp = df_pp.drop(cat_cols_onc, axis=1)
        df_pp = pd.concat([df_pp, df_enc], axis=1)

    dictionary_metadata = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "datetime_cols": datetime_cols,
        "cat_cols_onc": cat_cols_onc
    }

    json_object = json.dumps(dictionary_metadata, indent=4)
    with open("data_pre_process_models/demo_metadata.json", "w") as outfile:
        outfile.write(json_object)

    pickle.dump(imp_mean, open('data_pre_process_models/demo_imp_mean.pkl', 'wb'))
    pickle.dump(imp_mode, open('data_pre_process_models/demo_imp_mode.pkl', 'wb'))
    pickle.dump(oh_enc, open('data_pre_process_models/demo_oh_enc.pkl', 'wb'))

    return df_pp


def process_demo_test_data(df):
    df = df.dropna(subset=['CustomerID'])

    # DEMOGRAPHICS
    if ('InsuredGender' in df.columns):
        df['InsuredGender'] = np.where(df['InsuredGender'] == 'NA', np.nan, df['InsuredGender'])

    if ('InsuredZipCode' in df.columns):
        df = df.drop(['InsuredZipCode'], axis=1)
    if ('Country' in df.columns):
        df = df.drop(['Country'], axis=1)

    with open('data_pre_process_models/demo_metadata.json', 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
        dictionary_metadata = dict(json_object)
    num_cols = dictionary_metadata['num_cols']
    cat_cols = dictionary_metadata['cat_cols']
    datetime_cols = dictionary_metadata['datetime_cols']
    cat_cols_onc = dictionary_metadata['cat_cols_onc']
    print("num_cols.  :", num_cols)
    print("cat_cols.  :", cat_cols)
    print("datetime_cols.  :", datetime_cols)
    print("cat_cols_onc.  :", cat_cols_onc)

    imp_mean = pickle.load(open('data_pre_process_models/demo_imp_mean.pkl', 'rb'))
    imp_mode = pickle.load(open('data_pre_process_models/demo_imp_mode.pkl', 'rb'))

    df_num = df[num_cols]
    df_cat = df[cat_cols]
    df_datetime = df[datetime_cols]

    if (len(num_cols)) > 0:
        df_num = pd.DataFrame(imp_mean.transform(df_num), columns=df_num.columns)
    if (len(cat_cols)) > 0:
        df_cat = pd.DataFrame(imp_mode.transform(df_cat), columns=df_cat.columns)

    df_pp = pd.concat([df_cat, df_num, df_datetime], axis=1)

    oh_enc = pickle.load(open('data_pre_process_models/demo_oh_enc.pkl', 'rb'))

    if (len(cat_cols_onc) > 0):
        print("\n cat_cols_onc. ::", cat_cols_onc)
        df_enc = pd.DataFrame(oh_enc.transform(df_pp[cat_cols_onc]).toarray(),
                              columns=oh_enc.get_feature_names(cat_cols_onc))
        df_pp = df_pp.drop(cat_cols_onc, axis=1)
        df_pp = pd.concat([df_pp, df_enc], axis=1)

    return df_pp



def PP_all_train_data():
    print("DEMO DATA ")
    df_train_demo = pd.read_csv('TrainData/Train_Demographics.csv')
    print("BEFORE PP ", df_train_demo.shape)
    df_train_demo_pp = process_demo_train_data(df_train_demo)
    print("AFTER PP ", df_train_demo_pp.shape)

    print("CLAIM DATA ")
    df_train_claim = pd.read_csv('TrainData/Train_Claim.csv')
    print("BEFORE PP ", df_train_claim.shape)
    df_train_claim_pp = process_claim_train_data(df_train_claim)
    print("AFTER PP ", df_train_claim_pp.shape)

    print("POLICY DATA ")
    df_train_policy = pd.read_csv('TrainData/Train_Policy.csv')
    print("BEFORE PP ", df_train_policy.shape)
    df_train_policy_pp = process_policy_train_data(df_train_policy)
    print("AFTER PP ", df_train_policy_pp.shape)

    print("VEHICLE DATA ")
    df_train_veh = pd.read_csv('../TrainData/Train_Vehicle.csv')
    print("BEFORE PP ", df_train_veh.shape)
    df_train_veh_pp = process_vehicle_train_data(df_train_veh)
    print("AFTER PP ", df_train_veh_pp.shape)

    train_target = pd.read_csv('TrainData/Traindata_with_Target.csv')
    train_target = train_target.dropna(subset=['CustomerID'])
    print("TRAIN DATA WITH TARGET", train_target.shape)
    le = LabelEncoder()
    train_target['ReportedFraud'] = le.fit_transform(train_target['ReportedFraud'])
    pickle.dump(le, open('data_pre_process_models/train_target_label_enc.pkl', 'wb'))

    df_train_merged = train_target.merge(df_train_demo_pp, how='inner', on=['CustomerID']) \
        .merge(df_train_policy_pp, how='inner', on=['CustomerID']) \
        .merge(df_train_claim_pp, how='inner', on=['CustomerID']) \
        .merge(df_train_veh_pp, how='inner', on=['CustomerID'])

    df_train_merged['no_days_incident_vehicleYOM'] = (df_train_merged['DateOfIncident']
                                                      - df_train_merged['VehicleAttributeDetails_VehicleYOM']).dt.days
    df_train_merged['no_days_incident_PolicyCoverage'] = (df_train_merged['DateOfIncident']
                                                          - df_train_merged['DateOfPolicyCoverage']).dt.days
    df_train_merged = df_train_merged.drop(['DateOfIncident',
                                            'VehicleAttributeDetails_VehicleYOM', 'DateOfPolicyCoverage'], axis=1)
    df_train_merged = df_train_merged.query("no_days_incident_vehicleYOM>=0 and no_days_incident_PolicyCoverage>=0")

    print("MERGING ALL DATASETS")
    print(df_train_merged.shape)
    if 'CustomerID' in df_train_merged.columns:
        df_train_merged=df_train_merged.drop(['CustomerID'], axis=1)
    df_train_merged.to_csv('TrainData/TrainData_merged_all.csv')



def PP_all_test_data():
    print("DEMO DATA ")
    df_test_demo = pd.read_csv('TestData/Test_Demographics.csv')
    print("BEFORE PP ", df_test_demo.shape)
    df_test_demo_pp = process_demo_test_data(df_test_demo)
    print("AFTER PP ", df_test_demo_pp.shape)

    print("CLAIM DATA ")
    df_test_claim = pd.read_csv('TestData/Test_Claim.csv')
    print("BEFORE PP ", df_test_claim.shape)
    df_test_claim_pp = process_claim_test_data(df_test_claim)
    print("AFTER PP ", df_test_claim_pp.shape)

    print("POLICY DATA ")
    df_test_policy=pd.read_csv('TestData/Test_Policy.csv')
    print("BEFORE PP ", df_test_policy.shape)
    df_test_policy_pp = process_policy_test_data(df_test_policy)
    print("AFTER PP ", df_test_policy_pp.shape)

    print("VEHICLE DATA ")
    df_test_veh = pd.read_csv('TestData/Test_Vehicle.csv')
    print("BEFORE PP ", df_test_veh.shape)
    df_test_veh_pp=process_vehicle_test_data(df_test_veh)
    print("AFTER PP ", df_test_veh_pp.shape)

    test_target = pd.read_csv('TestData/Test.csv')
    test_target = test_target.dropna(subset=['CustomerID'])
    print("TEST DATA WITH TARGET", test_target.shape)

    df_test_merged = test_target.merge(df_test_demo_pp, how='inner', on=['CustomerID']) \
        .merge(df_test_policy_pp, how='inner', on=['CustomerID']) \
        .merge(df_test_claim_pp, how='inner', on=['CustomerID']) \
        .merge(df_test_veh_pp, how='inner', on=['CustomerID'])

    df_test_merged['no_days_incident_vehicleYOM'] = (df_test_merged['DateOfIncident']
                                                     - df_test_merged['VehicleAttributeDetails_VehicleYOM']).dt.days
    df_test_merged['no_days_incident_PolicyCoverage'] = (df_test_merged['DateOfIncident']
                                                         - df_test_merged['DateOfPolicyCoverage']).dt.days
    df_test_merged = df_test_merged.drop(['DateOfIncident',
                                          'VehicleAttributeDetails_VehicleYOM', 'DateOfPolicyCoverage'], axis=1)
    df_test_merged = df_test_merged.query("no_days_incident_vehicleYOM>=0 and no_days_incident_PolicyCoverage>=0")

    print("MERGING ALL TEST DATASETS")
    print(df_test_merged.shape)
    df_test_merged.to_csv('TestData/TestData_merged_all.csv')


def PP_all_test_data_custom(df_test_demo,df_test_claim,df_test_policy,df_test_veh):
    print("DEMO DATA ")
    # df_test_demo = pd.read_csv('TestData/Test_Demographics.csv')
    print("BEFORE PP ", df_test_demo.shape)
    df_test_demo_pp = process_demo_test_data(df_test_demo)
    print("AFTER PP ", df_test_demo_pp.shape)

    print("CLAIM DATA ")
    # df_test_claim = pd.read_csv('TestData/Test_Claim.csv')
    print("BEFORE PP ", df_test_claim.shape)
    df_test_claim_pp = process_claim_test_data(df_test_claim)
    print("AFTER PP ", df_test_claim_pp.shape)

    print("POLICY DATA ")
    # df_test_policy=pd.read_csv('TestData/Test_Policy.csv')
    print("BEFORE PP ", df_test_policy.shape)
    df_test_policy_pp = process_policy_test_data(df_test_policy)
    print("AFTER PP ", df_test_policy_pp.shape)

    print("VEHICLE DATA ")
    # df_test_veh = pd.read_csv('TestData/Test_Vehicle.csv')
    print("BEFORE PP ", df_test_veh.shape)
    df_test_veh_pp=process_vehicle_test_data(df_test_veh)
    print("AFTER PP ", df_test_veh_pp.shape)


    df_test_merged = df_test_demo_pp.merge(df_test_policy_pp, how='inner', on=['CustomerID']) \
        .merge(df_test_claim_pp, how='inner', on=['CustomerID']) \
        .merge(df_test_veh_pp, how='inner', on=['CustomerID'])

    try:
        df_test_merged['no_days_incident_vehicleYOM'] = (df_test_merged['DateOfIncident']
                                                         - df_test_merged['VehicleAttributeDetails_VehicleYOM']).dt.days
    except:
        df_test_merged['no_days_incident_vehicleYOM'] = np.nan
    try:
        df_test_merged['no_days_incident_PolicyCoverage'] = (df_test_merged['DateOfIncident']
                                                         - df_test_merged['DateOfPolicyCoverage']).dt.days
    except:
        df_test_merged['no_days_incident_PolicyCoverage'] = np.nan
    df_test_merged = df_test_merged.drop(['DateOfIncident',
                                          'VehicleAttributeDetails_VehicleYOM', 'DateOfPolicyCoverage'], axis=1)
    df_test_merged = df_test_merged.query("no_days_incident_vehicleYOM>=0 and no_days_incident_PolicyCoverage>=0")

    print("MERGING ALL TEST DATASETS")
    print(df_test_merged.shape)
    df_test_merged.to_csv('TestData/TestData_merged_all_custom.csv')

def train_model():
    df_train = pd.read_csv("TrainData/TrainData_merged_all.csv")
    print(df_train.shape)
    if 'Unnamed: 0' in df_train.columns:
        df_train = df_train.drop(['Unnamed: 0'],axis=1)
    print(df_train.columns)
    X_train_all = df_train.drop(['ReportedFraud'], axis=1)
    Y_train_all = df_train[['ReportedFraud']]

    dictionary_metadata = {
        "cols_model_input": X_train_all.columns.tolist()
    }

    json_object = json.dumps(dictionary_metadata, indent=4)
    with open("data_pre_process_models/model_cols_metadata.json", "w") as outfile:
        outfile.write(json_object)



    X_train, X_test, y_train, y_test = train_test_split(X_train_all, Y_train_all,
                                                        test_size=0.33, random_state=42,
                                                        stratify=Y_train_all)

    xgb_model = xgb.XGBClassifier()

    # brute force scan for all parameters, here are the tricks
    # usually max_depth is 6,7,8
    # learning rate is around 0.05, but small changes may make big diff
    # tuning min_child_weight subsample colsample_bytree can have
    # much fun of fighting against overfit
    # n_estimators is how many round of boosting
    # finally, ensemble xgboost with multiple seeds may reduce variance
    parameters = {'nthread': [4],  # when use hyperthread, xgboost may become slower
                  'objective': ['binary:logistic'],
                  'learning_rate': [0.05],  # so called `eta` value
                  'max_depth': [6, 8],
                  'min_child_weight': [11],
                  'silent': [1],
                  'subsample': [0.8],
                  'colsample_bytree': [0.7],
                  'n_estimators': [100, 200, 400],  # number of trees, change it to 1000 for better results
                  'missing': [-999],
                  'seed': [1337]}

    clf = GridSearchCV(xgb_model, parameters, n_jobs=5,
                       cv=3,
                       scoring='accuracy',
                       verbose=2, refit=True)

    clf.fit(X_train, y_train)

    print(" best_params_ ", clf.best_params_)
    print(" best_score_ ", clf.best_score_)
    final_model = clf.best_estimator_
    print(" final_model ",final_model)
    cols_model = X_train.columns.tolist()
    X_test = X_test[cols_model]
    print(X_test.shape)
    y_preds_val = final_model.predict(X_test)
    print("\nModel Report on validation data ")
    print("Accuracy : %.4g" % accuracy_score(y_test, y_preds_val))
    print("AUC Score (Train): %f" % roc_auc_score(y_test, y_preds_val))
    print(classification_report(y_test, y_preds_val))

    # feat_imp = pd.Series(final_model.feature_importances_, X_train.columns).sort_values(ascending=False)
    # feat_imp.head(15).plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')

    pickle.dump(clf, open('model_insurance_fraud.pkl', 'wb'))


def get_preds(df_predict):
    if 'VehicleAttributeDetails_VehicleModel_RSX' not in df_predict.columns:
        df_predict['VehicleAttributeDetails_VehicleModel_RSX']=0
    with open('data_pre_process_models/model_cols_metadata.json', 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
        dictionary_metadata = dict(json_object)
    cols_model_input = dictionary_metadata['cols_model_input']
    model = pickle.load(open('model_insurance_fraud.pkl', 'rb'))
    preds_new_data = model.predict(df_predict[cols_model_input])
    df_predict['predictions'] = preds_new_data

    return df_predict





# train_model()
# PP_all_train_data()
# PP_all_test_data()