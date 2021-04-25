FROM continuumio/anaconda3:2019.03

RUN apt update && apt install -y gcc g++

WORKDIR /opt/ml
COPY requirements.txt /opt/ml
RUN pip install -r requirements.txt
