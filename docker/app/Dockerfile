FROM spark:4.0.2-scala2.13-java17-python3-ubuntu

USER root

RUN pip3 install --no-cache-dir numpy

WORKDIR /opt/app

COPY src/train.py /opt/app/train.py
COPY src/predict.py /opt/app/predict.py
COPY src/visualize.py /opt/app/visualize.py
COPY src/benchmark.py /opt/app/benchmark.py

USER spark