# 기업과제 3 NLU - 문장 유사도 계산 (STS)
10팀

팀원 : 남준우, 조성빈, 권예환, 강상훈, 장수림

# Install

필요 라이브러리 설치

    pip install -r requirements.txt
    
Model 다운로드

Link : https://drive.google.com/file/d/1uAZWx0NHuznyusg_D06d8gFXId8uC_DJ/view?usp=sharing

model.zip 파일 다운로드 후 API 구성 폴더 내에서 압축 해제 진행

# Model

모델의 경우 sentence_transformers의 klue/roberta-base을 이용하여 SetnenceBert로 학습하였습니다.
# API Framework

API는 FastAPI를 이용하여 API를 제작하였습니다.

# Run API

    uvicorn main:app --reload --host=127.0.0.1 --port=8001
    
# Information

 - Method = POST
 - Require Parameter = json { "sentence1" : str , "sentence2" : str }
 - Response = json  {"sentence1": sentence1, "sentence2": sentence2,
                     "similarity_score" : float, "binary label": 1 or 0}
                     
# Usage
-PostMan 사용 시 호출 URI

    http://server-ip:port/similarity
    or
    http://dns:port/similarity
    
    
# Result
## Swagger
![api](https://user-images.githubusercontent.com/60643348/159421366-45f73e52-773e-4203-a67d-e7a4d68aa4c6.png)

## Postman
![api2](https://user-images.githubusercontent.com/60643348/159421378-a0c99797-c1e2-4e5a-81c0-00514324b05a.png)

# Reference
Sentence Transformer : https://github.com/UKPLab/sentence-transformers

FastAPI : https://fastapi.tiangolo.com/ko/
