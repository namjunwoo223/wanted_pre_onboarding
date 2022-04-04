# 원티드 AI 모의면접 사전과제
## API Code Download
URI : https://drive.google.com/file/d/1vWZm3sTTLeLiT9cAVBRGYstGB3ku0M9K/view?usp=sharing

## Docker를 이용한 환경설치
   docker run -it -v dir:mount-dir -p 8001:8001 --name wanted_api wanted_api:1.0
   docker build --tag wanted_api:1.0 .
   
## Model
SKTBrain에서 제공해주는 KoBert를 이용하여 학습 진행하였습니다. 단순하게 pooler를 뽑아줍니다.

## API Framework
API Framework는 Django rest Framework를 이용하여 제작하였습니다.

## RUN
    gunicorn --bind 0.0.0.0:8001 api.wsgi:application -t 60

## Database
사전과제 요청 중 데이터베이스 이용을 원하였기 때문에 DAO.py를 통한 데이터베이스 접근을 하여 응답받은 파라미터들을 데이터베이스에 저장해두며
동시에 pooler를 통하여 나온 벡터데이터를 같이 저장합니다.

데이터베이스의 경우 당시 MariaDB를 이용하여 사용하였습니다.

## 호출방법
호출 uri는
inference, siminference 두 가지가 존재하며 inference의 경우 KoBert의 예측 값을 return해주고
siminference의 경우 유사도를 비교하여 비슷한 문단을 return 해줍니다.
