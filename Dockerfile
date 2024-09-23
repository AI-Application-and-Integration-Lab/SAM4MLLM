FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# install git
RUN apt-get update && apt-get install -y git

# cd into LLaVA-NeXT and pip install
COPY ./LLaVA-NeXT ./LLaVA-NeXT
RUN cd LLaVA-NeXT && pip install -e ".[train]"

# install python dependencies
RUN pip install --no-cache-dir orjson wandb
RUN pip install --no-cache-dir transformers==4.42.4 peft==0.11.0 lightning==2.3.3 \
deepspeed==0.14.4 trl==0.9.6
RUN pip install flash-attn==2.5.9.post1 --no-build-isolation

# install path of transformers /opt/conda/lib/python3.10/site-packages/transformers/__init__.py
# comment out logits = logits.float()  in /opt/conda/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py
RUN sed -i 's/logits = logits.float()/#logits = logits.float()/' /opt/conda/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py

# set environment variables
ENV PYTHONPATH=/app

# set default command
CMD ["python", "sam4mllm_train.py"]