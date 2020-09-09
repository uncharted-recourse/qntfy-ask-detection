FROM ubuntu:bionic

ARG BRANCH_NAME=__UNSET__
ENV BRANCH_NAME=${BRANCH_NAME}
ENV XDG_CACHE_HOME=/cache/
# Primitive system requirements
RUN apt-get -qq update -y \
  && apt-get -qq install -y \
  && apt-get install -y unzip zip \
  build-essential \
  python-pip \
  python3.6 \
  python3-pip \
  python3-dev \
  && rm -rf /var/lib/apt/lists/*

# Configure python3.6 at `python`
RUN ln -s -f /usr/bin/python3.6 /usr/bin/python \
    && ln -s -f /usr/bin/pip3 /usr/bin/pip

RUN pip install dumb-init

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt
COPY . .

RUN unzip /models/cnn_classifier.h5.zip -d /models
RUN chmod -R 777 /root/
RUN cp -r /root/nltk_data /nltk_data
ENTRYPOINT ["/usr/local/bin/dumb-init", "--"]
CMD ["python3", "/src/ask_detection_server.py"]
