from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# Define your FastAPI app
app = FastAPI()

# Load the trained model
model = load("models/GBC.joblib")
imputer = load("models/KNNimputer.joblib")


class InputData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    duration: int
    campaign: int
    previous: int
    poutcome: str
        
        
class OutputData(BaseModel):
    # Define the output data structure
    prediction: int
    probability: float

def encode_data(data):
    # Create a DataFrame
    df = pd.DataFrame(data, index=[0])

    # List of columns for one-hot encoding
    one_hot_cols = ['job','default', 'housing', 'loan', 'poutcome']

    # Mapping for categorical encoding - also for the ones which are not in the dataset to keep the spacing.
    category_mapping_day_of_week = {'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4}
    category_mapping_month = {'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5, 'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11}
    category_mapping_education = {'illiterate': 0, 'basic.4y': 1, 'basic.6y': 2, 'basic.9y': 3, 'high.school': 4, 'professional.course': 5,'university.degree': 6, 'unknown': "unknown"}
    category_mapping_marital = {'single': 0, 'married': 1, 'divorced': 2, 'unknown': "unknown"}
    category_mapping_contact = {'telephone': 0, 'cellular': 1}

    colums_excluding_y = ['age', 'marital', 'education', 'contact', 'month', 'day_of_week',
       'duration', 'campaign', 'previous', 'job_admin.',
       'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
       'job_management', 'job_retired', 'job_self-employed', 'job_services',
       'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
       'default_no', 'default_unknown', 'default_yes', 'housing_no',
       'housing_unknown', 'housing_yes', 'loan_no', 'loan_unknown', 'loan_yes',
       'poutcome_failure', 'poutcome_nonexistent', 'poutcome_success',
       'season']

    # Perform one-hot encoding on data
    encoded_df = pd.get_dummies(df, columns=one_hot_cols)
    fill_values = {"marital": "unknown", "education": "unknown"}
    encoded_df = encoded_df.reindex(columns=colums_excluding_y, fill_value=0)
    encoded_df = encoded_df.fillna(fill_values)

    # Perform categorical encoding on  data
    encoded_df['day_of_week'] = encoded_df['day_of_week'].map(category_mapping_day_of_week)
    encoded_df['month'] = encoded_df['month'].map(category_mapping_month)
    encoded_df['education'] = encoded_df['education'].map(category_mapping_education)
    encoded_df['marital'] = encoded_df['marital'].map(category_mapping_marital)
    encoded_df['contact'] = encoded_df['contact'].map(category_mapping_contact)

    return encoded_df

def impute_df(df):
    imputed_array = imputer.transform(df.replace('unknown', np.nan))
    imputed_df = pd.DataFrame(imputed_array, columns=df.columns)
    season_mapping = {4:0,5:0,6:0,7:0, 0:1,1:1,2:1,3:1,8:1,9:1,10:1,11:1}
    imputed_df['season'] = imputed_df['month'].map(season_mapping)
    return imputed_df

@app.post("/predict", response_model=OutputData)
def predict(data: InputData):
    # Preprocess the input data (if required)
    encoded_df = encode_data(dict(data))
    imputed_df = impute_df(encoded_df)
    y = model.predict(imputed_df)[0]
    probability = model.predict_proba(imputed_df)[:,1][0]
    response = OutputData(prediction=y, probability=probability)
    return response