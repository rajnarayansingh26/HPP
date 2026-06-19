from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import json

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="California Housing Predictor")

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Templates
# -----------------------------
templates = Jinja2Templates(directory="templates")

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("model.pkl")

# -----------------------------
# Cache
# -----------------------------
prediction_cache = {}

# -----------------------------
# Input Schema
# -----------------------------
class HouseInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# -----------------------------
# Home Page
# -----------------------------
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
async def predict(data: HouseInput):

    cache_key = json.dumps(
        data.model_dump(),
        sort_keys=True
    )

    # Cache Hit
    if cache_key in prediction_cache:
        return {
            "prediction": prediction_cache[cache_key],
            "source": "cache"
        }

    features = [[
        data.MedInc,
        data.HouseAge,
        data.AveRooms,
        data.AveBedrms,
        data.Population,
        data.AveOccup,
        data.Latitude,
        data.Longitude
    ]]

    prediction = float(model.predict(features)[0])

    prediction_cache[cache_key] = prediction

    return {
        "prediction": prediction,
        "source": "model"
    }

# -----------------------------
# Cache Information
# -----------------------------
@app.get("/cache")
async def cache_info():
    return {
        "cached_requests": len(prediction_cache)
    }

# -----------------------------
# Health Check
# -----------------------------
@app.get("/health")
async def health():
    return {
        "status": "running"
    }