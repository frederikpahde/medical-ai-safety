#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ICON_REPO="$SCRIPT_DIR/../ICON_decomposition"

if [ ! -d "$ICON_REPO" ]; then
    echo "WARNING: ICON repo not found at $ICON_REPO" >&2
    exit 1
fi

model=vgg16
for dir in "$SCRIPT_DIR"/model_checkpoints/${model}_*/; do
    ckpt_name=$(basename "$dir" | sed "s/_/-/g")
    best_ckpt=$(ls "${dir}"*_best.pth 2>/dev/null | head -1)
    if [ -z "$best_ckpt" ]; then
        echo "WARNING: no best checkpoint found in $dir, skipping" >&2
        continue
    fi
    dest="$ICON_REPO/experiments/melanoma/output/isic_y-lbl-NV__${model}/trial_0/checkpoints/${ckpt_name}.pth"
    mkdir -p "$(dirname "$dest")"
    cp "$best_ckpt" "$dest"
    echo "Copied $(basename "$best_ckpt") -> $dest"
done