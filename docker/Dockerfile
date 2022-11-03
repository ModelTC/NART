FROM registry.sensetime.com/nart/nvidia/cuda:11.0-devel-ubuntu16.04

LABEL maintainer="compile-link@sensetime.com"

RUN apt-get update && apt-get install -y \
        build-essential \
        cmake \
        make \
        unzip \
        wget \
        python3-dev python3-pip \
        && rm -rf /var/lib/apt/lists/*

ENV PYENV_ROOT=/opt/pyenv \
    PATH=/opt/pyenv/shims:/opt/pyenv/bin:$PATH \
    CAFFEPATH=/opt/python/caffe

ENV PYTHONPATH=/opt/python:$PYTHONPATH
ENV LD_LIBRARY_PATH=$CAFFEPATH:/usr/local/lib:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y \
    git \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    python-numpy \
    && rm -rf /var/lib/apt/lists/*

RUN git config --global http.proxy 'http://172.16.1.135:3128' \
    && git config --global https.proxy 'http://172.16.1.135:3128'

RUN git clone --depth 1 https://github.com/pyenv/pyenv.git /opt/pyenv \
    && eval "$(pyenv init -)" \
    && env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.6.4 \
    && pyenv global 3.6.4

RUN pip3 install -U pip -i https://repo.sensetime.com/repository/pypi/simple/
RUN pip3 config set global.index-url https://repo.sensetime.com/repository/pypi/simple/

# tensorrt7.1 && cudnn8.0
RUN cd /opt/ \
    && wget -q http://somewhere/somehow/cuda11.0/TensorRT-7.1.3.4.Ubuntu-16.04.x86_64-gnu.cuda-11.0.cudnn8.0.tar.gz \
    && tar -xvf TensorRT-7.1.3.4.Ubuntu-16.04.x86_64-gnu.cuda-11.0.cudnn8.0.tar.gz \
    && cd /opt/TensorRT-7.1.3.4/python \
    && pip install tensorrt-7.1.3.4-cp36-none-linux_x86_64.whl \
    && rm /opt/TensorRT-7.1.3.4.Ubuntu-16.04.x86_64-gnu.cuda-11.0.cudnn8.0.tar.gz

ENV LD_LIBRARY_PATH=/opt/TensorRT-7.1.3.4/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN mkdir /dep \
    && cd /dep \
    && wget -q http://somewhere/somehow/cuda11.0/cudnn-11.0-linux-x64-v8.0.3.33.tar.gz \
    && tar -xvf cudnn-11.0-linux-x64-v8.0.3.33.tar.gz \
    && cd /dep/cuda \
    && cp include/* /usr/local/cuda/include/ -r \
    && cp lib64/* /usr/local/cuda/lib64/ -ard \
    && cd / \
    && rm /dep -r

RUN pip install opencv-python==3.4.0.12 onnx==1.7.0 packaging numpy setuptools wheel -U
RUN pip install -i http://spring.sensetime.com/pypi/ --trusted-host spring.sensetime.com dipoorlet==0.0.11 pycuda==2020.1

COPY python/dist python/dist
RUN pip install python/dist/nart*.whl \
    && rm -rf python/dist
