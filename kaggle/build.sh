#!/bin/sh
# Builds the Kaggle push folders for the pilot and the main run from theseus.py.
# Push with the Kaggle CLI, for example:
#   kaggle kernels push -p kaggle/push/theseus-pilot --accelerator NvidiaTeslaT4
set -e
cd "$(dirname "$0")"
for mode in pilot main; do
  d="push/theseus-$mode"
  mkdir -p "$d"
  sed "s/^MODE = \"pilot\"/MODE = \"$mode\"/" theseus.py > "$d/theseus.py"
  cat > "$d/kernel-metadata.json" <<EOF
{
 "id": "pragyaangaur/theseus-$mode",
 "title": "theseus $mode",
 "code_file": "theseus.py",
 "language": "python",
 "kernel_type": "script",
 "is_private": true,
 "enable_gpu": true,
 "enable_internet": true,
 "dataset_sources": [],
 "competition_sources": [],
 "kernel_sources": [],
 "model_sources": []
}
EOF
done
grep -n '^MODE' push/*/theseus.py
