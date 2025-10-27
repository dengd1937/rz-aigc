FROM python:3.12-slim

WORKDIR /app

# 安装依赖（如需）
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# 克隆 VLMEvalKit
RUN git clone https://github.com/open-compass/VLMEvalKit.git .
RUN pip install --no-cache-dir -r requirements.txt

# 复制替换脚本并执行
RUN cd /app/VLMEvalKit && \
COPY replace_model_paths.py . && \
COPY .env . && \
RUN python replace_model_paths.py

# 清理
RUN rm replace_model_paths.py

# 默认挂载点
ENV MODEL_PATH=/models

CMD ["bash"]