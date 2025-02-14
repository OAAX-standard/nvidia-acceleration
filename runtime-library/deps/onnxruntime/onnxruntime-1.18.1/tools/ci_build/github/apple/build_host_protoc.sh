#!/bin/bash

# Note: This script is intended to be called from a macOS pipeline to build the host protoc
# See tools/ci_build/github/azure-pipelines/mac-ios-ci-pipeline.yml
# The host_protoc can be found as $PROTOC_INSTALL_PATH/bin/protoc

set -e

if [ $# -ne 3 ]
then
    echo "Usage: ${0} <repo_root_path> <host_protoc_build_path> <host_protoc_install_path>"
    exit 1
fi

set -x

ORT_REPO_ROOT=$1
PROTOC_BUILD_PATH=$2
PROTOC_INSTALL_PATH=$3

pushd .
mkdir -p "$PROTOC_BUILD_PATH"
cd "$PROTOC_BUILD_PATH"
DEP_FILE_PATH="$ORT_REPO_ROOT/cmake/deps.txt"
PATCH_FILE_PATH="$ORT_REPO_ROOT/cmake/patches/protobuf/protobuf_cmake.patch"
protobuf_url=$(grep '^protobuf' "$DEP_FILE_PATH" | cut -d ';' -f 2 | sed 's/\.zip$/\.tar.gz/')
curl -sSL --retry 5 --retry-delay 10 --create-dirs --fail -L -o protobuf_src.tar.gz "$protobuf_url"
tar -zxf protobuf_src.tar.gz --strip=1
patch --binary --ignore-whitespace -p1 < "$PATCH_FILE_PATH"
# The second 'cmake' is a folder name
cmake cmake \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -Dprotobuf_BUILD_TESTS=OFF \
    -Dprotobuf_WITH_ZLIB_DEFAULT=OFF \
    -Dprotobuf_BUILD_SHARED_LIBS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    "-DCMAKE_INSTALL_PREFIX=$PROTOC_INSTALL_PATH"
make -j $(getconf _NPROCESSORS_ONLN)
make install
popd
