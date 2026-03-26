set -e

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <input-zip-path> <output-directory>" >&2
  exit 1
fi

conversion_toolchain --input-path "$1" --output-dir "$2"
