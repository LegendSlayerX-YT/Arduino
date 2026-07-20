#!/bin/sh

set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SOURCE_DIR="$ROOT_DIR/custom_libraries"
TARGET_DIR="$ROOT_DIR/libraries"

if [ ! -d "$SOURCE_DIR" ]; then
  echo "Source directory not found: $SOURCE_DIR" >&2
  exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
  echo "Target directory not found: $TARGET_DIR" >&2
  exit 1
fi

found_any=0

for source_path in "$SOURCE_DIR"/*; do
  if [ ! -d "$source_path" ]; then
    continue
  fi

  found_any=1
  name=$(basename "$source_path")
  target_path="$TARGET_DIR/$name"

  if [ -e "$target_path" ] && [ ! -L "$target_path" ]; then
    echo "Skipping $name: $target_path already exists and is not a symlink." >&2
    continue
  fi

  ln -sfn "$source_path" "$target_path"
  echo "Linked $target_path -> $source_path"
done

if [ "$found_any" -eq 0 ]; then
  echo "No subdirectories found in $SOURCE_DIR"
fi
