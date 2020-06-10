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

RUN apt-get install -y ffmpeg
RUN apt-get install -y libxkbcommon-x11-0

RUN rm -f /workspace/requirements.txt /workspace/p.patch
WORKDIR /workspace/rtvc

RUN pip install flask
RUN pip install google-colab

EXPOSE 8888
EXPOSE 10050
