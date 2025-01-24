FROM python:3.9-slim

RUN apt-get update && apt-get install -y gnugo && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    six \
    numpy \
    matplotlib \
    keras \
    tensorflow \
    h5py \
    flask

WORKDIR /app

CMD ["/bin/bash"]