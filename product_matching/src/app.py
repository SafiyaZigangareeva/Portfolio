from fastapi import FastAPI
import uvicorn
from typing import Union
import faiss
import numpy as np
from joblib import load
from preprocessing import preprocess_features

app = FastAPI()
dims = 72
faiss_index = None
model = None
mapper = {}


def parse_string(vec: str) -> list[float]:
    """
    1.23,6.7 -> [1.23 6.7]
    :param vec:
    :return:
    """
    l = vec.split(",")
    if len(l) != dims:
        return None
    return [float(el) for el in l]


@app.on_event("startup")
def start():
    global faiss_index
    global model
    global scaler

    scaler = load('../scaler.pkl')
    n_cells = 5
    faiss_index = load('../index_ip.pkl')


@app.get("/")
def main() -> dict:
    return {"status": "OK", "message": "Hello, world!"}


@app.get("/knn")
def match(item: Union[str, None] = None) -> dict:
    global faiss_index
    if item is None:
        return {"status": "fail", "message": "No input data"}

    vec = parse_string(item)
    vec = preprocess_features(vec)
    knn, idx = faiss_index.search(np.ascontiguousarray(vec), k = 5)

    base_index = load('../base_index.pkl')
    return {"status": "OK", "data": [base_index[int(ind)] for ind in idx[0]]}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8031)