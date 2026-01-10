from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pickle
import numpy as np
import os

app = FastAPI(
    title="Real Estate Price Prediction API",
    description="Predict house prices in Bangalore using ML models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

AVAILABLE_MODELS = {
    "linear_regression": "linear_regression.pkl",
    "decision_tree": "decision_tree.pkl",
    "random_forest": "random_forest.pkl",
    "gradient_boosting": "gradient_boosting.pkl"
}

LOCATIONS = [
    "Whitefield", "Sarjapur  Road", "Electronic City", "Marathahalli", 
    "Raja Rajeshwari Nagar", "Haralur Road", "Hennur Road", "Bannerghatta Road",
    "Uttarahalli", "Thanisandra", "Electronic City Phase II", "Hebbal",
    "7th Phase JP Nagar", "Yelahanka", "Kanakpura Road", "KR Puram", "Sarjapur",
    "Rajaji Nagar", "Kasavanhalli", "Bellandur", "Begur Road", "Banashankari",
    "Kothanur", "Hormavu", "Harlur", "Akshaya Nagar", "Jakkur",
    "Electronics City Phase 1", "Varthur", "Chandapura", "HSR Layout", "Hennur",
    "Ramamurthy Nagar", "Ramagondanahalli", "Kaggadasapura", "Kundalahalli",
    "Koramangala", "Hulimavu", "Budigere", "Hoodi", "Malleshwaram", "Hegde Nagar",
    "8th Phase JP Nagar", "Gottigere", "JP Nagar", "Yeshwanthpur", "Channasandra",
    "Bisuvanahalli", "Vittasandra", "Indira Nagar", "Vijayanagar", "Kengeri",
    "Brookefield", "Sahakara Nagar", "Hosa Road", "Old Airport Road", "Bommasandra",
    "Balagere", "Green Glen Layout", "Old Madras Road", "Rachenahalli", "Panathur",
    "Kudlu Gate", "Thigalarapalya", "Ambedkar Nagar", "Jigani", "Yelahanka New Town",
    "Talaghattapura", "Mysore Road", "Kadugodi", "Frazer Town", "Dodda Nekkundi",
    "Devanahalli", "Kanakapura", "Attibele", "Anekal", "Lakshminarayana Pura",
    "Nagarbhavi", "Ananth Nagar", "5th Phase JP Nagar", "TC Palaya", "CV Raman Nagar",
    "Kengeri Satellite Town", "Kudlu", "Jalahalli", "Subramanyapura", "Bhoganhalli",
    "Doddathoguru", "Kalena Agrahara", "Horamavu Agara", "Vidyaranyapura",
    "BTM 2nd Stage", "Hebbal Kempapura", "Hosur Road", "Horamavu Banaswadi",
    "Domlur", "Mahadevpura", "Tumkur Road"
]

AREA_TYPES = ["Super built-up  Area", "Built-up  Area", "Plot  Area"]


class PropertyInput(BaseModel):
    bath: float
    balcony: float
    total_sqft: float
    bhk: int
    location: str
    area_type: str
    ready_to_move: bool = True


class PredictionRequest(BaseModel):
    property_data: PropertyInput
    model_name: str = "gradient_boosting"


class PredictionResponse(BaseModel):
    predicted_price: float
    model_used: str
    price_per_sqft: float


def load_model(model_name: str):
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not found. Available: {list(AVAILABLE_MODELS.keys())}")
    
    model_path = os.path.join(MODELS_DIR, AVAILABLE_MODELS[model_name])
    if not os.path.exists(model_path):
        raise HTTPException(status_code=500, detail=f"Model file not found: {model_path}")
    
    with open(model_path, "rb") as f:
        return pickle.load(f)


def prepare_features(data: PropertyInput) -> np.ndarray:
    features = []
    
    features.append(data.bath)
    features.append(data.balcony)
    features.append(data.total_sqft)
    features.append(data.bhk)
    
    for at in AREA_TYPES:
        features.append(1 if data.area_type == at else 0)
    
    features.append(1 if data.ready_to_move else 0)
    
    for loc in LOCATIONS:
        features.append(1 if data.location == loc else 0)
    
    features.append(data.bath / data.bhk)
    features.append(data.balcony / data.bhk)
    features.append(data.total_sqft / data.bhk)
    features.append(data.bath + data.balcony)
    features.append(np.log1p(data.total_sqft))
    features.append(data.total_sqft * data.bhk)
    features.append(data.total_sqft * data.bath)
    features.append(data.bhk * data.bath)
    features.append(data.total_sqft ** 2)
    features.append(data.bhk ** 2)
    
    return np.array(features).reshape(1, -1)


@app.get("/")
def root():
    return {
        "message": "Real Estate Price Prediction API",
        "endpoints": {
            "/predict": "POST - Dynamic model prediction",
            "/predict/linear_regression": "POST - Linear Regression prediction",
            "/predict/decision_tree": "POST - Decision Tree prediction",
            "/predict/random_forest": "POST - Random Forest prediction",
            "/predict/gradient_boosting": "POST - Gradient Boosting prediction",
            "/models": "GET - List available models",
            "/locations": "GET - List available locations"
        }
    }


@app.get("/models")
def get_models():
    return {"available_models": list(AVAILABLE_MODELS.keys())}


@app.get("/locations")
def get_locations():
    return {"locations": LOCATIONS, "count": len(LOCATIONS)}


@app.get("/area_types")
def get_area_types():
    return {"area_types": AREA_TYPES}


@app.post("/predict", response_model=PredictionResponse)
def predict_dynamic(request: PredictionRequest):
    model = load_model(request.model_name)
    features = prepare_features(request.property_data)
    prediction = model.predict(features)[0]
    
    return PredictionResponse(
        predicted_price=round(prediction, 2),
        model_used=request.model_name,
        price_per_sqft=round(prediction * 100000 / request.property_data.total_sqft, 2)
    )


@app.post("/predict/linear_regression", response_model=PredictionResponse)
def predict_linear_regression(property_data: PropertyInput):
    model = load_model("linear_regression")
    features = prepare_features(property_data)
    prediction = model.predict(features)[0]
    
    return PredictionResponse(
        predicted_price=round(prediction, 2),
        model_used="linear_regression",
        price_per_sqft=round(prediction * 100000 / property_data.total_sqft, 2)
    )


@app.post("/predict/decision_tree", response_model=PredictionResponse)
def predict_decision_tree(property_data: PropertyInput):
    model = load_model("decision_tree")
    features = prepare_features(property_data)
    prediction = model.predict(features)[0]
    
    return PredictionResponse(
        predicted_price=round(prediction, 2),
        model_used="decision_tree",
        price_per_sqft=round(prediction * 100000 / property_data.total_sqft, 2)
    )


@app.post("/predict/random_forest", response_model=PredictionResponse)
def predict_random_forest(property_data: PropertyInput):
    model = load_model("random_forest")
    features = prepare_features(property_data)
    prediction = model.predict(features)[0]
    
    return PredictionResponse(
        predicted_price=round(prediction, 2),
        model_used="random_forest",
        price_per_sqft=round(prediction * 100000 / property_data.total_sqft, 2)
    )


@app.post("/predict/gradient_boosting", response_model=PredictionResponse)
def predict_gradient_boosting(property_data: PropertyInput):
    model = load_model("gradient_boosting")
    features = prepare_features(property_data)
    prediction = model.predict(features)[0]
    
    return PredictionResponse(
        predicted_price=round(prediction, 2),
        model_used="gradient_boosting",
        price_per_sqft=round(prediction * 100000 / property_data.total_sqft, 2)
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
