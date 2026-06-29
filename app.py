from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from sklearn.datasets import load_iris

# Load model
model = joblib.load("iris_model.pkl")

# Load iris dataset
iris = load_iris()
target_names = iris.target_names

# FastAPI app
app = FastAPI()

# Input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Home endpoint
@app.get("/")
def home():
    return {"message": "Welcome to Iris Flower Classification API"}

# Prediction endpoint
@app.post("/predict")
def predict(data: IrisInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(features)

    return {
        "prediction": target_names[prediction[0]]
    }