ARG DEBIAN_FRONTEND=noninteractive
FROM python:3.8.16 as base

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN apt-get update -qq -y --fix-missing \
  && apt-get install -qq -y gcc g++ curl make xxd wget apt-utils software-properties-common git autoconf \
    automake libtool python3 unzip cmake build-essential protobuf-compiler libprotoc-dev > /dev/null

WORKDIR /app

COPY VERSION /app/VERSION
COPY onnxruntime/conversion-toolchain/pyproject.toml onnxruntime/conversion-toolchain/uv.lock /app/
COPY onnxruntime/conversion-toolchain/conversion_toolchain /app/conversion_toolchain
COPY onnxruntime/conversion-toolchain/scripts/ /app/scripts

RUN pip install --no-cache-dir uv && \
    uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["bash", "/app/scripts/convert.sh"]
