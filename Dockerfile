ARG BASE_IMG="common"
FROM python:3.8 as common

# Non-volative layers
WORKDIR /root
RUN git clone https://github.com/huggingface/neuralcoref.git

COPY pip.conf /root/.pip/pip.conf

WORKDIR neuralcoref/
RUN pip install -r requirements.txt && \
    pip install -e . && \
    pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz

COPY requirements.txt /root/requirements.txt
RUN pip install -r /root/requirements.txt

FROM ${BASE_IMG} as am_combiner

WORKDIR /am_combiner
COPY . /am_combiner/

ENV PYTHONPATH=/am_combiner/

ENTRYPOINT ["python", "am_combiner/__main__.py"]

FROM ${BASE_IMG} as test_image

# Packages for graph visualizations
RUN apt update && apt install -y graphviz graphviz-dev

WORKDIR /am_combiner
COPY requirements-dev.txt requirements-dev.txt
RUN pip install -r requirements-dev.txt

# makes jupyter more usable
RUN jupyter contrib nbextension install --user
RUN pip install jupyter_nbextensions_configurator
RUN jupyter nbextensions_configurator enable --user

COPY . /am_combiner/

ENV PYTHONPATH=/am_combiner/
