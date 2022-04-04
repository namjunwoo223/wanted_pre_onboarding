FROM python:3.8-slim-buster

RUN apt-get update -y

RUN apt-get -y install nano git build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev
RUN apt-get -y install python3-dev default-libmysqlclient-dev build-essential libgl1-mesa-glx
RUN pip install mysqlclient

RUN pip install transformers git+https://git@github.com/SKTBrain/KoBERT.git@master sentencepiece gluonnlp mxnet konlpy

RUN pip install torch==1.7.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install cython

RUN pip install opencv-python django djangorestframework markdown django-filter boto3 django-storages awscli django-cors-headers sklearn gunicorn uvicorn pymysql

EXPOSE 8001