import os

from fastapi import FastAPI
from starlette.responses import JSONResponse
from pydantic import BaseModel

import json

import math
import pandas as pd
from datetime import datetime

import torch
import transformers
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, models

import inference

app = FastAPI() #FastAPI 실행

if torch.cuda.is_available(): #현재 로컬의 device상태를 확인하고 device 설정
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


embedding_model = models.Transformer( 
    model_name_or_path = "klue/roberta-base",
    max_seq_length = 256,
    do_lower_case = False
    )


pooling_model = models.Pooling(
    embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False,
)

model = SentenceTransformer(f"{os.getcwd()}\\model", #현재 위치에 있는 model 폴더안의 모델을 불러옴
                            modules = [embedding_model, pooling_model], 
                            cache_folder = f"{os.getcwd()}\\model", 
                            device = device)

class Sentence(BaseModel): #Json Recieve 선언
    sentence1: str
    sentence2: str


@app.get("/")
def hello(): # test api
    return {"Hello": "World"} 

@app.post("/similarity") #Cosine similarity 계산
async def similarity(Sentence: Sentence):
    sentences = dict(Sentence)
    embeddings1 = model.encode(sentences['sentence1'])
    embeddings2 = model.encode(sentences['sentence2'])

    cos_score = inference.cal_cos_score(embeddings1, embeddings2)
    
    res_dict = {"sentence1": sentences['sentence1'], 
                "sentence2": sentences['sentence2'] ,
                "similarity_score" : float(cos_score) * 5, # 처음 학습시 스케일링을 진행하였고 그 스케일링을 풀어주는 역할
                "binary label": 1 if float(cos_score) > 0.6 else 0} # 점수가 0.6 이상이면 positive(1) 아니면 negative(0)

    return JSONResponse(res_dict)
