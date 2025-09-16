# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from sklearn.ensemble import RandomForestClassifier
import psycopg2
from contextlib import contextmanager
import uuid
from typing import Optional
import traceback
from datetime import datetime
import os
from urllib.parse import urlparse

# ==============================
# Database config (Postgres)
# ==============================
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("❌ DATABASE_URL chưa được cấu hình trong Environment Variables")

url = urlparse(DATABASE_URL)

DB_CONFIG = {
    "dbname": url.path[1:],  # bỏ dấu "/" đầu
    "user": url.username,
    "password": url.password,
    "host": url.hostname,
    "port": url.port,
}

def init_database():
    """Chỉ tạo bảng nếu chưa có, không tạo database mới"""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # training_data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_data (
            id SERIAL PRIMARY KEY,
            height INTEGER NOT NULL,
            weight INTEGER NOT NULL,
            gender VARCHAR(20) NOT NULL,
            item_type VARCHAR(20) NOT NULL,
            actual_size VARCHAR(20) NOT NULL,
            foot_length FLOAT,
            chest_size FLOAT,
            waist_size FLOAT,
            hip_size FLOAT,
            brand VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # predictions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id UUID PRIMARY KEY,
            height INTEGER NOT NULL,
            weight INTEGER NOT NULL,
            gender VARCHAR(20) NOT NULL,
            item_type VARCHAR(20) NOT NULL,
            predicted_size VARCHAR(20) NOT NULL,
            confidence_score VARCHAR(20),
            body_type VARCHAR(20),
            user_feedback VARCHAR(50),
            actual_size VARCHAR(20),
            is_correct BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            feedback_at TIMESTAMP
        )
    ''')

    # modstatus
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS modstatus (
            id SERIAL PRIMARY KEY,
            prediction_id UUID REFERENCES predictions(id),
            user_feedback VARCHAR(50) NOT NULL,
            actual_size VARCHAR(20),
            feedback_type VARCHAR(20) DEFAULT 'user_correction',
            predicted_size VARCHAR(20),
            is_correct BOOLEAN,
            height INTEGER,
            weight INTEGER,
            gender VARCHAR(20),
            item_type VARCHAR(20),
            body_type VARCHAR(20),
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()

@contextmanager
def get_db():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
    finally:
        conn.close()

# ==============================
# Training dataset mẫu
# ==============================
X_male_shirt = [[165,55],[170,65],[175,75],[180,85]]
y_male_shirt = ["S","M","L","XL"]

X_female_shirt = [[155,45],[160,50],[165,55],[170,60]]
y_female_shirt = ["S","M","L","XL"]

X_male_pants = [[165,55],[170,65],[175,75],[180,85],[185,90]]
y_male_pants = ["28","30","32","34","36"]

X_female_pants = [[155,45],[160,50],[165,55],[170,60]]
y_female_pants = ["26","27","28","29"]

models = {}

def train_models():
    models['male_shirt'] = RandomForestClassifier().fit(X_male_shirt, y_male_shirt)
    models['female_shirt'] = RandomForestClassifier().fit(X_female_shirt, y_female_shirt)
    models['male_pants'] = RandomForestClassifier().fit(X_male_pants, y_male_pants)
    models['female_pants'] = RandomForestClassifier().fit(X_female_pants, y_female_pants)

# ==============================
# Pydantic Schemas
# ==============================
class SizeInput(BaseModel):
    height: int
    weight: int
    gender: str
    item_type: str
    body_type: Optional[str] = "Bình thường"

    @validator('gender')
    def validate_gender(cls, v):
        if v.lower() not in ['male','female']:
            raise ValueError("Gender phải là male hoặc female")
        return v.lower()

    @validator('item_type')
    def validate_item_type(cls, v):
        if v.lower() not in ['shirt','pants']:
            raise ValueError("Item phải là shirt hoặc pants")
        return v.lower()

    @validator('body_type')
    def validate_body_type(cls, v):
        if v not in ['Gầy', 'Bình thường', 'Đầy đặn']:
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
app = FastAPI(title="Size Prediction API with PostgreSQL + Body Type + ModStatus")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    init_database()
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

        size_order_shirt = ["S","M","L","XL","XXL"]
        size_order_pants = ["26","27","28","29","30","32","34","36"]
        size_order = size_order_shirt if data.item_type=="shirt" else size_order_pants

        idx = size_order.index(predicted_size) if predicted_size in size_order else 0
        bmi = data.weight / ((data.height / 100) ** 2)

        if data.item_type=="shirt":
            if bmi > 25 and idx < len(size_order)-1:
                idx += 1
            elif bmi < 18.5 and idx > 0:
                idx -= 1

        if data.body_type=="Gầy" and idx>0:
            idx -= 1
        elif data.body_type=="Đầy đặn" and idx < len(size_order)-1:
            idx +=1

        predicted_size = size_order[idx]

        prediction_id = str(uuid.uuid4())
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions (id,height,weight,gender,item_type,predicted_size,confidence_score,body_type)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            ''', (prediction_id, data.height, data.weight, data.gender, data.item_type, predicted_size, "medium", data.body_type))
            conn.commit()

        return {"prediction_id": prediction_id, "predicted_size": predicted_size, "bmi": round(bmi,1)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-training-data")
def add_training_data(data: TrainingData):
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO training_data (height,weight,gender,item_type,actual_size)
                VALUES (%s,%s,%s,%s,%s)
            ''', (data.height, data.weight, data.gender, data.item_type, data.actual_size))
            conn.commit()
        return {"message": "Đã thêm dữ liệu training", "data": data.dict()}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback/{prediction_id}")
def feedback(prediction_id: str, fb: FeedbackData):
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions WHERE id=%s", (prediction_id,))
            prediction = cursor.fetchone()
            if prediction is None:
                raise HTTPException(status_code=404, detail="Không tìm thấy prediction")

            cursor.execute('''
                UPDATE predictions
                SET user_feedback=%s, actual_size=%s, is_correct=%s, feedback_at=NOW()
                WHERE id=%s
            ''', (fb.feedback, fb.actual_size, fb.feedback=="correct", prediction_id))

            cursor.execute('''
                INSERT INTO modstatus (
                    prediction_id, user_feedback, actual_size, predicted_size,
                    is_correct, height, weight, gender, item_type, body_type, notes
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (
                prediction_id,
                fb.feedback,
                fb.actual_size,
                prediction[5],
                fb.feedback == "correct",
                prediction[1],
                prediction[2],
                prediction[3],
                prediction[4],
                prediction[7],
                fb.notes
            ))
            conn.commit()

        return {"message": "Feedback đã được lưu vào modstatus", "prediction_id": prediction_id}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/modstatus")
def get_modstatus(limit: int = 50):
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, prediction_id, user_feedback, actual_size, predicted_size,
                       is_correct, height, weight, gender, item_type, body_type, 
                       notes, created_at
                FROM modstatus
                ORDER BY created_at DESC
                LIMIT %s
            ''', (limit,))
            results = cursor.fetchall()

            columns = ['id', 'prediction_id', 'user_feedback', 'actual_size', 'predicted_size',
                      'is_correct', 'height', 'weight', 'gender', 'item_type', 'body_type',
                      'notes', 'created_at']

            feedback_list = []
            for row in results:
                feedback_dict = dict(zip(columns, row))
                if feedback_dict['created_at']:
                    feedback_dict['created_at'] = feedback_dict['created_at'].isoformat()
                feedback_list.append(feedback_dict)

        return {"feedback_count": len(feedback_list), "feedbacks": feedback_list}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Size Prediction API with PostgreSQL + Body Type + ModStatus running 🚀"}
