from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from typing import List, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Nutrilance ML API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class UserProfile(BaseModel):
    age: int
    gender: str  # "male" or "female"
    weight: float
    height: float
    activity: str  # "rendah", "sedang", "berat"
    goal: str  # "turun", "naik", "maintain"

class RecommendationRequest(BaseModel):
    target_calories: float
    meal_type: str
    preferences: Optional[List[str]] = []

class PredictionResponse(BaseModel):
    target_calories: float
    nutrient_category: str
    bmi: float
    confidence: float

# Global variables for models
calorie_model = None
category_model = None
scaler = None
encoders = None
nutrition_data = None

@app.on_event("startup")
async def load_models():
    global calorie_model, category_model, scaler, encoders, nutrition_data
    
    try:
        # Load models (you'll upload these files)
        calorie_model = joblib.load("models/calorie_model.joblib")
        category_model = joblib.load("models/category_model.joblib")
        scaler = joblib.load("models/scaler.joblib")
        encoders = joblib.load("models/encoders.joblib")
        
        # Load nutrition data
        nutrition_data = pd.read_csv("data/nutrition_data.csv")
        
        logger.info("Models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        # Fallback: create dummy models for testing
        create_fallback_models()

def create_fallback_models():
    """Create simple fallback models if main models fail to load"""
    global calorie_model, category_model, scaler, encoders, nutrition_data
    
    logger.warning("Using fallback models")
    
    # Create dummy data for testing
    nutrition_data = pd.DataFrame({
        'ingredient': ['Nasi Putih', 'Ayam Dada', 'Brokoli', 'Telur'],
        'calories': [130, 165, 34, 155],
        'protein': [2.7, 31, 2.8, 13],
        'fat': [0.3, 3.6, 0.4, 11],
        'carbohydrates': [28, 0, 7, 1.1],
        'fiber': [0.4, 0, 2.6, 0]
    })

def calculate_bmr(age: int, gender: str, weight: float, height: float) -> float:
    """Calculate Basal Metabolic Rate using Mifflin-St Jeor Equation"""
    if gender.lower() in ['male', 'laki-laki']:
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    return bmr

def get_activity_factor(activity: str) -> float:
    """Get activity factor for calorie calculation"""
    factors = {
        'rendah': 1.2,
        'sedang': 1.55,
        'berat': 1.725,
        'sangat_berat': 1.9
    }
    return factors.get(activity.lower(), 1.2)

def calculate_target_calories(user_profile: UserProfile) -> tuple:
    """Calculate target calories and determine nutrient category"""
    try:
        if calorie_model and scaler and encoders:
            # Use ML model if available
            return predict_with_ml_model(user_profile)
        else:
            # Use fallback calculation
            return predict_with_fallback(user_profile)
    except Exception as e:
        logger.error(f"Error in ML prediction: {e}")
        return predict_with_fallback(user_profile)

def predict_with_ml_model(user_profile: UserProfile) -> tuple:
    """Predict using trained ML models"""
    # Encode categorical variables
    gender_encoded = encoders['gender'].transform([user_profile.gender])[0]
    activity_encoded = encoders['activity'].transform([user_profile.activity])[0]
    goal_encoded = encoders['goal'].transform([user_profile.goal])[0]
    
    # Prepare features
    features = np.array([[
        user_profile.age,
        gender_encoded,
        user_profile.weight,
        user_profile.height,
        activity_encoded,
        goal_encoded
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    target_calories = calorie_model.predict(features_scaled)[0]
    nutrient_category_encoded = category_model.predict(features_scaled)[0]
    nutrient_category = encoders['nutrient_category'].inverse_transform([nutrient_category_encoded])[0]
    
    return target_calories, nutrient_category, 0.95

def predict_with_fallback(user_profile: UserProfile) -> tuple:
    """Fallback prediction using BMR calculation"""
    bmr = calculate_bmr(
        user_profile.age,
        user_profile.gender,
        user_profile.weight,
        user_profile.height
    )
    
    activity_factor = get_activity_factor(user_profile.activity)
    calories = bmr * activity_factor
    
    # Apply goal adjustments
    if user_profile.goal == "turun":
        calories -= 300
    elif user_profile.goal == "naik":
        calories += 300
    
    # Calculate BMI for category determination
    bmi = user_profile.weight / (user_profile.height / 100) ** 2
    
    # Determine nutrient category
    if calories > 2200:
        nutrient_category = "Tinggi"
    elif calories < 1800:
        nutrient_category = "Rendah"
    else:
        nutrient_category = "Sedang"
    
    return calories, nutrient_category, 0.75

@app.get("/")
async def root():
    return {
        "message": "Nutrilance ML API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": calorie_model is not None,
        "data_loaded": nutrition_data is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_calories(user_profile: UserProfile):
    """Predict target calories and nutrient category for user"""
    try:
        target_calories, nutrient_category, confidence = calculate_target_calories(user_profile)
        
        # Calculate BMI
        bmi = user_profile.weight / (user_profile.height / 100) ** 2
        
        return PredictionResponse(
            target_calories=round(target_calories),
            nutrient_category=nutrient_category,
            bmi=round(bmi, 2),
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/recommendations")
async def get_recommendations(request: RecommendationRequest):
    """Get food recommendations based on target calories and meal type"""
    try:
        if nutrition_data is None:
            raise HTTPException(status_code=500, detail="Nutrition data not loaded")
        
        # Meal calorie distribution
        meal_distribution = {
            'sarapan': 0.25,
            'makan_siang': 0.35,
            'makan_malam': 0.30,
            'snack': 0.10
        }
        
        target_meal_calories = request.target_calories * meal_distribution.get(request.meal_type, 0.25)
        
        # Calculate scores for each food item
        recommendations = []
        for _, row in nutrition_data.iterrows():
            score = calculate_nutrition_score(row, target_meal_calories)
            recommendations.append({
                'ingredient': row['ingredient'],
                'calories': row['calories'],
                'protein': row['protein'],
                'fat': row['fat'],
                'carbohydrates': row['carbohydrates'],
                'fiber': row.get('fiber', 0),
                'score': score
            })
        
        # Sort by score and return top 10
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            "recommendations": recommendations[:10],
            "meal_type": request.meal_type,
            "target_calories": target_meal_calories
        }
        
    except Exception as e:
        logger.error(f"Error in recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

def calculate_nutrition_score(food_row, target_calories):
    """Calculate nutrition score for a food item"""
    # Calorie score (closer to target is better)
    calorie_diff = abs(food_row['calories'] - target_calories)
    calorie_score = 1 / (1 + calorie_diff / 100)
    
    # Protein score (higher is better, up to a point)
    protein_score = min(food_row['protein'] / 30, 1)
    
    # Fiber score (higher is better)
    fiber_score = min(food_row.get('fiber', 0) / 10, 1)
    
    # Fat penalty (too much fat is bad)
    fat_penalty = 1 if food_row['fat'] <= 20 else 0.7
    
    # Composite score
    total_score = (
        calorie_score * 0.4 +
        protein_score * 0.3 +
        fiber_score * 0.3
    ) * fat_penalty
    
    return total_score

# main.py (sama seperti sebelumnya, tapi tambahkan di akhir:)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)