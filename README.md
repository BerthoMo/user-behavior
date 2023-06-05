# User Behavior Analysis 
A bank performed a marketing campaign to track and predict the user behaviour. In this repository, the user behavior is analyzed, predicted and an API is exposed for individual predictions.

## Table of Contents

- [Installation](#installation)
- [API Usage](#apiusage)
- [Project Structure](#project-structure)


## Installation
How to clone repository, install the environment and include source data

### Pull the repository
```
git clone https://github.com/BerthoMo/user-behavior.git
cd user-behavior 
```
### Install environment using conda and open jupyter notebook
I recommend to use Anaconda as a package manager: https://docs.anaconda.com/free/anaconda/install/

```
conda env create --name ml_training --file environment.yml 
conda activate ml_training 
jupyter-notebook 
```

### Optional: Download source data - needed if you want to run the notebooks
Download test_file.xlsx and train_file.xlsx and put it into the data/raw_data/ subfolder.

## API Usage

### Start APP Server

The server can be started without retraining, since the models are included in the repository. After having installed all dependencies, run

```
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Retrieving predictions

Using for example Postman, use HTTP POST requests to get predictions on data:

```
Endpoint: http://localhost:8000/predict
```

Example Body Payload y="yes"

```
{
  "age": 21,
  "job": "job.student",
  "marital": "married",
  "education": "high.school",
  "default": "no",
  "housing": "yes",
  "loan": "no",
  "contact": "telephone",
  "month": "sep",
  "day_of_week": "wed",
  "duration": 400,
  "campaign": 4,
  "previous": 0,
  "poutcome": "nonexistent"
}
```


Example Body Payload y="no"
```
{
  "age": 30,
  "job": "job.student",
  "marital": "married",
  "education": "high.school",
  "default": "no",
  "housing": "yes",
  "loan": "no",
  "contact": "telephone",
  "month": "sep",
  "day_of_week": "wed",
  "duration": 100,
  "campaign": 4,
  "previous": 0,
  "poutcome": "nonexistent"
}
```
Output: 
```

{
    "prediction": int,
    "probability": float
}

```

## Project Structure

The two important notebooks are:
- EDA & Preprocessing.ipynb
    - contains feature mappings, unknown variable imputations and other interesting findings about the data
- Model Training & Selection, Feature Importance and Test Submission.ipynb


```
.
├── EDA & Preprocessing.ipynb
├── Hyperparameter Tuning using Grid Search.ipynb
├── Model Training & Selection, Feature Importance and Test Submission.ipynb
├── README.md
├── app.py
├── data
│   ├── predictions
│   │   └── test_data_predicted.csv
│   ├── preprocessed_data
│   │   ├── imputed_data.csv
│   │   └── imputed_test_data.csv
│   └── raw_data
│       ├── test_file.xlsx
│       └── train_file.xlsx
├── environment.yml
└── models
    ├── GBC.joblib
    └── KNNimputer.joblib
```