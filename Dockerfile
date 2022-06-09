FROM python:3.8-buster

COPY api /api
COPY requirements.txt /requirements.txt
COPY XPerts /XPerts
COPY MANIFEST.in /MANIFEST.in
COPY model.h5 /model.h5

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
