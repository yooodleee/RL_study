FROM python:3.6

RUN apt-get -y update && apt-get -y install ffmpeg
# RUN apt-get -y update && apt-get -y install git wget python-env python3-dev libopenmpi-dev python-pip zliblg-dev cmake python-opencv

ENV CODE_DIR /root/src

COPY . $CODE_DIR/baselines
WORKDIR $CORD_DIR/baselines

# Clean up pycache and pyc files
RUN rm -rf __pacache__ && \
    find . -name "*.pyc" - delete && \
    pip install tensorflow && \
    pip install -e .[test]


CMD /bin/bash