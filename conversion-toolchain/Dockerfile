ARG DEBIAN_FRONTEND=noninteractive
FROM nvcr.io/nvidia/tensorrt:26.03-py3

WORKDIR /app

COPY VERSION /app/VERSION
COPY conversion-toolchain/pyproject.toml conversion-toolchain/uv.lock /app/
COPY conversion-toolchain/conversion_toolchain /app/conversion_toolchain
COPY conversion-toolchain/scripts/ /app/scripts

RUN pip install --no-cache-dir uv && \
    uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["bash", "/app/scripts/convert.sh"]
