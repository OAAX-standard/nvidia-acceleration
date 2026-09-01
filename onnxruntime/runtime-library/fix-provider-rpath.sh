#!/bin/bash
# Rewrites the RUNPATH of the prebuilt ORT libs vendored into bin/ to $ORIGIN.
# ORT's own build bakes in an absolute, build-machine-only RUNPATH for its CUDA
# provider (e.g. /opt/cudnn-...-archive/lib), which doesn't exist once the .so
# is deployed elsewhere. $ORIGIN is intentionally kept out of CMakeLists.txt's
# add_custom_command text: Make interprets a bare `$O` in a recipe line as its
# own single-letter variable reference, mangling "$ORIGIN" into "RIGIN".
set -e

bin_dir="$1"

patchelf --set-rpath '$ORIGIN' "${bin_dir}"/*.so
