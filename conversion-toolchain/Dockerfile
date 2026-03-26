ARG DEBIAN_FRONTEND=noninteractive
FROM nvcr.io/nvidia/tensorrt:26.03-py3

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY setup.py /app/setup.py
COPY conversion_toolchain /app/conversion_toolchain
COPY scripts/ /app/scripts

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["bash", "/app/scripts/convert.sh"]
