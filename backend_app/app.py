import os
import pickle
import traceback
from typing import List
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


class InputData(BaseModel):
    features: List[float]


directoryPath = "models"

svmModel = "SVM_model.pkl"
lrModel = "LR_model.pkl"
rfModel = "RF_model.pkl"

svmPath = os.path.join(directoryPath, svmModel)
lrPath = os.path.join(directoryPath, lrModel)
rfPath = os.path.join(directoryPath, rfModel)

# Load models
try:
    with open(lrPath, 'rb') as f:
        logistic_regression = joblib.load(f)
    with open(rfPath, "rb") as f:
        random_forest = pickle.load(f)
    with open(svmPath, "rb") as f:
        svm = pickle.load(f)
    with open(lrPath, "rb") as f:
        logistic_regression = pickle.load(f)
except Exception as e:
    print(f"Error loading models: {e}")
    logistic_regression, random_forest, svm = None, None, None


mapping_dict = {0: "malignant", 1: "benign"}

@app.get('/')
def reed_root():
    return {'message': 'Breast Cancer API'}

@app.post("/predict/lr")
async def predict_model1(input_data: InputData):
    try:
        data = np.array(input_data.features).reshape(1, -1)
        prediction = logistic_regression.predict(data)
        result = f"The patient has {mapping_dict[prediction[0]]} tumor"
        return {"respone": result}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/rf")
async def predict_model2(input_data: InputData):
    try:
        data = np.array(input_data.features).reshape(1, -1)
        prediction = random_forest.predict(data)
        result = f"The patient has {mapping_dict[prediction[0]]} tumor"
        return {"respone": result}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/svm")
async def predict_model3(input_data: InputData):
    try:
        data = np.array(input_data.features).reshape(1, -1)
        prediction = svm.predict(data)
        result = f"The patient has {mapping_dict[prediction[0]]} tumor"
        return {"respone": result}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
