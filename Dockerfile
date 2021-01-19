FROM python
WORKDIR /usr/src/app
COPY . /usr/src/app
RUN pip install -r requirements.txt
CMD ["python", "/usr/src/app/app.py"]

