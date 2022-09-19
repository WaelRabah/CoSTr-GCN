from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI, exceptions
from data_loaders.graph import Graph
from test_online import load_model, labels
import torch
import json
from fastapi.middleware.cors import CORSMiddleware


graph = torch.from_numpy(Graph(layout="SHREC21", strategy="distance").A)
model=load_model(graph)

class Window(BaseModel):
    frames: list
    


with open('thresholds.json',mode="r") as f:
    thresholds=json.load(f)

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def read_root():
    return "Welcome to the STr-GCN online recognition server"

@app.get('/load_model/')
def load_model_():
    model=load_model(graph)
    return "Model loaded ..."

@app.post("/predict_window",status_code=200)
def predict_window(window: Window):
    global model
    if model==None :
        model=load_model(graph)


    w=torch.tensor(window.frames,dtype=torch.float)
    print(w.shape)
    if len(w.shape) != 3 :
        return exceptions.HTTPException(400, "Invalid window shape")
    if w.shape[1] != 20 or w.shape[2] != 3 :
        return exceptions.HTTPException(400, "Invalid window shape: each frame should contain 20 joints with their 3d coordinates")
    w=w.unsqueeze(0)
    score = model(w)
    prob=torch.nn.functional.softmax(score, dim=-1) 

    score_list_labels= torch.argmax(prob, dim=-1)
    # if prob[0][score_list_labels[0].item()] < thresholds[str(score_list_labels[0].item())]['threshold_avg']:
    #     return {"label":labels[0],"idx":0}
    
    return {"label":labels[score_list_labels[0].item()],"idx":score_list_labels[0].item()}