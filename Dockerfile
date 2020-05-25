ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.03-py3
FROM ${FROM_IMAGE_NAME}

ADD requirements.txt /workspace/requirements.txt
ADD p.patch /workspace/p.patch
WORKDIR /workspace

RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

RUN pip install --upgrade "jupyter_http_over_ws>=0.0.7" && \
  jupyter serverextension enable --py jupyter_http_over_ws
RUN pip install ipywidgets && \
  jupyter nbextension enable --py widgetsnbextension

RUN apt update && apt install -y \
    libportaudio2 \
&& rm -rf /var/lib/apt/list/*

RUN patch -i p.patch \
  /opt/conda/lib/python3.6/site-packages/numba/typeconv/castgraph.py

RUN pip install google-colab
