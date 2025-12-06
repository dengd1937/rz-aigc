FROM python:3.10

# 设置环境变量
ENV DATASET_SOURCE=ModelScope

# 更新系统并安装基础依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        wget \
        unzip \
        ca-certificates \
        vim \
        htop && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 升级pip
RUN pip install --upgrade pip

# 克隆并安装OpenCompass
RUN git clone https://github.com/open-compass/opencompass.git /workspace/opencompass && \
    cd /workspace/opencompass && \
    pip install -e . && \
    pip install modelscope

# 下载核心数据集
RUN cd /workspace/opencompass && \
    wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip && \
    unzip OpenCompassData-core-20240207.zip

# 设置工作目录
WORKDIR /workspace/opencompass

# 默认命令
CMD ["/bin/bash"]