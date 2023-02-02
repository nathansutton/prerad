FROM python:3.9-buster

RUN \
    apt-get update && \
    apt-get -y upgrade && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN useradd --create-home app
WORKDIR /home/app

COPY requirements.txt /home/app/
COPY run.py /home/app/

RUN \
    chown app:app /home/app/requirements.txt && \
    chmod 0755 /home/app/requirements.txt && \
    chown app:app /home/app/run.py && \
    chmod 0755 /home/app/run.py

USER app

ENV VIRTUAL_ENV=/home/app/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN \
    pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["python", "run.py", "worker", "-l", "info"]