FROM nvidia/cuda:10.2-runtime-ubuntu18.04
COPY . /usr/app/
WORKDIR /usr/app/
RUN apt update && apt install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools
RUN pip3 -q install pip --upgrade
RUN pip3 install --no-cache-dir -r requirements.txt
EXPOSE 8000:8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]