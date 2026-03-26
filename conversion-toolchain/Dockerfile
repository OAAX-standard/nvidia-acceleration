ARG DEBIAN_FRONTEND=noninteractive
FROM nvcr.io/nvidia/tensorrt:26.03-py3

WORKDIR /app

COPY pyproject.toml uv.lock /app/
COPY conversion_toolchain /app/conversion_toolchain
COPY scripts/ /app/scripts

RUN pip install --no-cache-dir uv && \
    uv sync --frozen --no-dev

ENTRYPOINT ["bash", "/app/scripts/convert.sh"]
