#!/bin/sh
# build.sh <sketch.pde> <outdir>  -> compiles the sketch against the stubs into <outdir>
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PDE=$1; OUT=$2
NAME=$(basename "$PDE" .pde)
mkdir -p "$OUT"
python3 "$HERE/pde2java.py" "$PDE" "$NAME" "$OUT/$NAME.java"
javac -nowarn -d "$OUT" "$HERE"/processing/core/*.java "$HERE"/oscP5/*.java "$HERE"/netP5/*.java "$HERE/Runner.java" "$OUT/$NAME.java"
echo "compiled $NAME"
