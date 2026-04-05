#!/usr/bin/env bash
# Copy data/*.csv files from the source repo, preserving relative paths.

SRC=$1
DST=$2

find "$SRC" -path '*/data/*.csv' | while read -r file; do
    rel="${file#$SRC/}"
    dest="$DST/$rel"
    mkdir -p "$(dirname "$dest")"
    cp -v "$file" "$dest"
done
