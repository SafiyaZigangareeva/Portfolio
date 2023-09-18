from fastapi import FastAPI, Request
import uvicorn
from joblib import load
from preprocessing import preprocess_features
import pandas as pd
from pydantic import BaseModel

class Item(BaseModel):
    url: str
    date_app: str
    date_update: str
    note: str
    apartment_type: str
    neighborhood: str
    street: str
    house: str
    floor: str
    layout: str
    total_area: float
    living_area: float
    kitchen_area: float
    views: int

app = FastAPI()
model = None

def parse_data(query):
    """Преобразует запрос в датафрейм"""
    return pd.DataFrame(query, index=[0])

@app.on_event("startup")

def start():
    global model
    model = load('../best_model.pkl')

@app.get("/")
def main() -> dict:
    return {"status": "OK",
            "message": "Hello, world!"}

@app.post('/test')
async def test(request):
    return await request.body()

@app.post("/items")
async def create_item(item: Item):
    return item.model_dump()

@app.post("/eval")
def evaluation(item: Item):
    vec = parse_data(item.model_dump())
    vec = preprocess_features(vec)
    vec = model[0].transform(vec)
    result = model[1].predict(vec)[0]
    return {"status": "OK", "price": result}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8031)