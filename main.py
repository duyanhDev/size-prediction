from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from sklearn.ensemble import RandomForestClassifier
from typing import Optional
from datetime import datetime
import traceback
import uuid
import os
from pymongo import MongoClient
from dotenv import load_dotenv

# ==============================
# Load .env
# ==============================
load_dotenv()

# ==============================
# Database config (MongoDB Atlas)
# ==============================
MONGO_URL = os.getenv("MONGO_URL")

if not MONGO_URL:
    raise RuntimeError("❌ MONGO_URL chưa được cấu hình trong Environment Variables")

client = MongoClient(MONGO_URL)
db = client["size_predictions"]

# ==============================
# Training dataset mẫu
# ==============================
X_male_shirt = [[165, 55], [170, 65], [175, 75], [180, 85]]
y_male_shirt = ["S", "M", "L", "XL"]

X_female_shirt = [[155, 45], [160, 50], [165, 55], [170, 60]]
y_female_shirt = ["S", "M", "L", "XL"]

X_male_pants = [[165, 55], [170, 65], [175, 75], [180, 85], [185, 90]]
y_male_pants = ["28", "30", "32", "34", "36"]

X_female_pants = [[155, 45], [160, 50], [165, 55], [170, 60]]
y_female_pants = ["26", "27", "28", "29"]

models = {}

def train_models():
    models["male_shirt"] = RandomForestClassifier().fit(X_male_shirt, y_male_shirt)
    models["female_shirt"] = RandomForestClassifier().fit(X_female_shirt, y_female_shirt)
    models["male_pants"] = RandomForestClassifier().fit(X_male_pants, y_male_pants)
    models["female_pants"] = RandomForestClassifier().fit(X_female_pants, y_female_pants)

# ==============================
# Pydantic Schemas
# ==============================
class SizeInput(BaseModel):
    height: int
    weight: int
    gender: str
    item_type: str
    body_type: Optional[str] = "Bình thường"

    @validator("gender")
    def validate_gender(cls, v):
        if v.lower() not in ["male", "female"]:
            raise ValueError("Gender phải là male hoặc female")
        return v.lower()

    @validator("item_type")
    def validate_item_type(cls, v):
        if v.lower() not in ["shirt", "pants"]:
            raise ValueError("Item phải là shirt hoặc pants")
        return v.lower()

    @validator("body_type")
    def validate_body_type(cls, v):
        if v not in ["Gầy", "Bình thường", "Đầy đặn"]:
            raise ValueError("Body type phải là Gầy / Bình thường / Đầy đặn")
        return v

class TrainingData(BaseModel):
    height: int
    weight: int
    gender: str
    item_type: str
    actual_size: str

class FeedbackData(BaseModel):
    feedback: str
    actual_size: Optional[str] = None
    notes: Optional[str] = None

# ==============================
# FastAPI app
# ==============================
app = FastAPI(title="Size Prediction API with MongoDB + Body Type + ModStatus")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    train_models()

# ==============================
# Endpoints
# ==============================
@app.post("/predict")
def predict_size(data: SizeInput):
    try:
        features = [[data.height, data.weight]]
        model_key = f"{data.gender}_{data.item_type}"

        if model_key not in models:
            raise HTTPException(status_code=400, detail="Model không tồn tại")

        predicted_size = models[model_key].predict(features)[0]

        size_order_shirt = ["S", "M", "L", "XL", "XXL"]
        size_order_pants = ["26", "27", "28", "29", "30", "32", "34", "36"]
        size_order = size_order_shirt if data.item_type == "shirt" else size_order_pants

        idx = size_order.index(predicted_size) if predicted_size in size_order else 0
        bmi = data.weight / ((data.height / 100) ** 2)

        if data.item_type == "shirt":
            if bmi > 25 and idx < len(size_order) - 1:
                idx += 1
            elif bmi < 18.5 and idx > 0:
                idx -= 1

        if data.body_type == "Gầy" and idx > 0:
            idx -= 1
        elif data.body_type == "Đầy đặn" and idx < len(size_order) - 1:
            idx += 1

        predicted_size = size_order[idx]

        prediction_id = str(uuid.uuid4())
        db.predictions.insert_one({
            "id": prediction_id,
            "height": data.height,
            "weight": data.weight,
            "gender": data.gender,
            "item_type": data.item_type,
            "predicted_size": predicted_size,
            "confidence_score": "medium",
            "body_type": data.body_type,
            "created_at": datetime.utcnow()
        })

        return {"prediction_id": prediction_id, "predicted_size": predicted_size, "bmi": round(bmi, 1)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-training-data")
def add_training_data(data: TrainingData):
    try:
        db.training_data.insert_one({
            **data.dict(),
            "created_at": datetime.utcnow()
        })
        return {"message": "Đã thêm dữ liệu training", "data": data.dict()}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback/{prediction_id}")
def feedback(prediction_id: str, fb: FeedbackData):
    try:
        prediction = db.predictions.find_one({"id": prediction_id})
        if not prediction:
            raise HTTPException(status_code=404, detail="Không tìm thấy prediction")

        db.predictions.update_one(
            {"id": prediction_id},
            {"$set": {
                "user_feedback": fb.feedback,
                "actual_size": fb.actual_size,
                "is_correct": fb.feedback == "correct",
                "feedback_at": datetime.utcnow()
            }}
        )

        db.modstatus.insert_one({
            "prediction_id": prediction_id,
            "user_feedback": fb.feedback,
            "actual_size": fb.actual_size,
            "predicted_size": prediction["predicted_size"],
            "is_correct": fb.feedback == "correct",
            "height": prediction["height"],
            "weight": prediction["weight"],
            "gender": prediction["gender"],
            "item_type": prediction["item_type"],
            "body_type": prediction.get("body_type"),
            "notes": fb.notes,
            "created_at": datetime.utcnow()
        })

        return {"message": "Feedback đã được lưu vào modstatus", "prediction_id": prediction_id}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/modstatus")
def get_modstatus(limit: int = 50):
    try:
        results = db.modstatus.find().sort("created_at", -1).limit(limit)
        feedback_list = []
        for r in results:
            r["_id"] = str(r["_id"])  # convert ObjectId to string
            if "created_at" in r:
                r["created_at"] = r["created_at"].isoformat()
            feedback_list.append(r)

        return {"feedback_count": len(feedback_list), "feedbacks": feedback_list}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Size Prediction API with MongoDB + Body Type + ModStatus running 🚀"}
