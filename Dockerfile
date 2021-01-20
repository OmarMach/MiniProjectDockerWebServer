FROM python:3.8-slim-buster

WORKDIR /usr/src/app

COPY . .

RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc \libsndfile1 

RUN apt-get install libsndfile1

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 9999 

CMD ["python", "./app.py"]

