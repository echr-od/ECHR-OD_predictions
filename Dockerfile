FROM python:3.8

WORKDIR /tmp/echr_experiments/
COPY requirements.txt .

RUN python3 -m pip install --upgrade pip==20.3.3
RUN python3 -m pip install --no-cache-dir  -r requirements.txt

ENTRYPOINT ["./entrypoint.sh"]