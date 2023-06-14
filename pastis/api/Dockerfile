FROM python:3.10.6-buster

WORKDIR /prod

COPY requirements_prod.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install .

COPY pastis.pastis / pastis
COPY setup.py / setup.py
COPY models_output / models_output

CMD uvicorn pastis.api.fast:app --host 0.0.0.0 --port $PORT
