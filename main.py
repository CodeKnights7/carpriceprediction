from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import io
import os
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Car Price Predictor API")
# ------------------------------------
# CORS Setup (Allow All Frontend Apps)
# ------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Allow ALL frontend URLs
    allow_credentials=True,
    allow_methods=["*"],          # GET, POST, PUT, DELETE, OPTIONS
    allow_headers=["*"],          # Authorization, Content-Type, etc.
)

MODEL_PATH = "car_price_model.pkl"

# ---------------------------
# Training function
# ---------------------------
def train_model(df: pd.DataFrame):
    # Drop unwanted columns
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df = df.drop(columns=["car_name", "brand"], errors="ignore")

    # Features and target
    if "selling_price" not in df.columns:
        raise ValueError("Dataset must contain 'selling_price' column")

    X = df.drop("selling_price", axis=1)
    y = df["selling_price"]

    categorical_cols = X.select_dtypes(include=["object"]).columns
    numeric_cols = X.select_dtypes(exclude=["object"]).columns

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_cols),
            ("numeric", numeric_transformer, numeric_cols)
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
    ])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Save model
    joblib.dump(model, MODEL_PATH)

    return {"mae": mae, "r2": r2}


# ---------------------------
# API Endpoints
# ---------------------------

@app.get("/")
def root():
    return {"message": "Car Price Predictor API is running. Use /train to train and /predict to predict."}


@app.post("/train")
async def train(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        metrics = train_model(df)
        return {"message": "Model trained successfully", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Pydantic model for prediction input
class CarFeatures(BaseModel):
    # Example fields — adjust based on your dataset columns
    vehicle_age: int
    kilometers_driven: int
    seller_type: str
    fuel_type: str
    transmission: str
    mileage: float
    engine: int
    max_power: float
    seats: int


@app.post("/predict")
def predict(features: CarFeatures):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=404, detail="Model not trained yet. Call /train first.")

    model = joblib.load(MODEL_PATH)

    # Convert input to DataFrame
    df = pd.DataFrame([features.dict()])

    try:
        prediction = model.predict(df)[0]
        return {"predicted_price": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
