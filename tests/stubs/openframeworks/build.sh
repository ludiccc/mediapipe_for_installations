#!/bin/sh
# build.sh <example dir> <output binary>  -> compiles src/*.cpp against the stub headers
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$1; OUT=$2
mkdir -p "$(dirname "$OUT")"
g++ -std=c++17 -Wall -Wno-unused-variable -Wno-unused-but-set-variable -I"$HERE" "$SRC"/src/*.cpp -o "$OUT" -lpthread
echo "compiled $SRC"
