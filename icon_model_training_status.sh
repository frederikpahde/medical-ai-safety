#!/bin/bash

CHECKPOINTS_DIR="$(dirname "$0")/model_checkpoints"
TOTAL_EPOCHS=300
BAR_WIDTH=25

# ANSI colors
BOLD="\033[1m"
RESET="\033[0m"
DIM="\033[2m"
GREEN="\033[32m"
YELLOW="\033[33m"
CYAN="\033[36m"

progress_bar() {
    local done=$1
    local total=$2
    local filled=$(( done * BAR_WIDTH / total ))
    local empty=$(( BAR_WIDTH - filled ))
    printf "["
    printf "%${filled}s" | tr ' ' '#'
    printf "%${empty}s" | tr ' ' '-'
    printf "]"
}

fmt_diff() {
    # Print a signed float with color: green if >=0, yellow if negative
    local val=$1
    if [[ -z "$val" || "$val" == "N/A" ]]; then
        printf "  N/A  "
        return
    fi
    local sign
    sign=$(awk -v v="$val" 'BEGIN { printf "%+.4f", v }')
    if awk -v v="$val" 'BEGIN { exit (v >= 0) }'; then
        printf "${YELLOW}%s${RESET}" "$sign"
    else
        printf "${GREEN}%s${RESET}" "$sign"
    fi
}

diff_val() {
    local a=$1 b=$2
    if [[ -z "$a" || -z "$b" ]]; then echo ""; return; fi
    awk -v a="$a" -v b="$b" 'BEGIN { printf "%.4f", a - b }'
}

get_col() {
    # get_col <csv_row> <header_line> <col_name>
    local row="$1" header="$2" name="$3"
    local idx
    idx=$(echo "$header" | tr ',' '\n' | grep -n "^${name}$" | cut -d: -f1)
    if [[ -z "$idx" ]]; then echo ""; return; fi
    echo "$row" | cut -d',' -f"$idx"
}

print_metric_row() {
    local label="$1" acc="$2" clean_acc="$3" attack_acc="$4"
    local d1 d2 d3
    d1=$(diff_val "$acc"        "$clean_acc")   # acc - clean_acc
    d2=$(diff_val "$attack_acc" "$acc")          # attack_acc - acc
    d3=$(diff_val "$attack_acc" "$clean_acc")    # attack_acc - clean_acc

    local acc_str clean_str attack_str
    acc_str=$(  [[ -n "$acc"        ]] && printf "%.4f" "$acc"        || echo "  N/A ")
    clean_str=$(  [[ -n "$clean_acc" ]] && printf "%.4f" "$clean_acc" || echo "  N/A ")
    attack_str=$( [[ -n "$attack_acc" ]] && printf "%.4f" "$attack_acc" || echo "  N/A ")

    printf "  %-6s  acc=%-7s  clean=%-7s  Δ(acc-clean)=%-10s  attack=%-7s  Δ(atk-acc)=%-10s  Δ(atk-clean)=%-10s\n" \
        "$label" "$acc_str" "$clean_str" "$(fmt_diff "$d1")" "$attack_str" "$(fmt_diff "$d2")" "$(fmt_diff "$d3")"
}

# ── Header ──────────────────────────────────────────────────────────────
printf "\n${BOLD}=== Model Training Status ===${RESET}\n\n"
printf "${DIM}%-30s  %-35s  %-14s  %-14s${RESET}\n" "Model" "Progress" "Train loss" "Train acc"
printf "%s\n" "$(printf '%.0s-' {1..110})"

for log in "$CHECKPOINTS_DIR"/vgg16_microscope-*/training_logs.csv; do
    model=$(basename "$(dirname "$log")")

    if [[ ! -f "$log" ]]; then
        printf "%-30s  [not found]\n" "$model"
        continue
    fi

    header=$(head -1 "$log")
    last_line=$(tail -1 "$log")

    # Training progress
    current_epoch=$(echo "$last_line" | cut -d',' -f1)
    done_epochs=$(( current_epoch + 1 ))
    train_loss=$(get_col "$last_line" "$header" "train_loss")
    train_acc=$(get_col  "$last_line" "$header" "train_accuracy_")

    # Last row with val metrics
    last_val=$(awk -F',' -v h="$header" '
        NR==1{next}
        {for(i=1;i<=NF;i++) if($i!="") last=$0}
        END{print last}
    ' "$log")
    # Actually: find last row where val_accuracy_ is present
    val_col=$(echo "$header" | tr ',' '\n' | grep -n "^val_accuracy_$" | cut -d: -f1)
    last_val=$(awk -F',' -v c="$val_col" 'NR>1 && $c!="" {row=$0} END{print row}' "$log")

    val_acc=$(        get_col "$last_val" "$header" "val_accuracy_")
    val_clean_acc=$(  get_col "$last_val" "$header" "val_clean_accuracy_")
    val_attack_acc=$( get_col "$last_val" "$header" "val_MEL-attackedby-microscope_accuracy_")
    test_acc=$(       get_col "$last_val" "$header" "test_clean_accuracy_")  # reuse col below
    test_clean_acc=$( get_col "$last_val" "$header" "test_clean_accuracy_")
    test_attack_acc=$(get_col "$last_val" "$header" "test_MEL-attackedby-microscope_accuracy_")
    # test_accuracy_ (non-clean) doesn't seem to exist; use val_accuracy_ equivalent
    # Actually check if test_accuracy_ exists
    test_acc_col=$(echo "$header" | tr ',' '\n' | grep -n "^test_accuracy_$" | cut -d: -f1)
    if [[ -n "$test_acc_col" ]]; then
        test_acc=$(get_col "$last_val" "$header" "test_accuracy_")
    else
        test_acc=""
    fi

    mod_time=$(stat -c '%y' "$log" | cut -d'.' -f1)

    # Progress bar line
    bar=$(progress_bar "$done_epochs" "$TOTAL_EPOCHS")
    train_loss_str=$( [[ -n "$train_loss" ]] && printf "%.4f" "$train_loss" || echo "N/A")
    train_acc_str=$(  [[ -n "$train_acc"  ]] && printf "%.4f" "$train_acc"  || echo "N/A")

    printf "${BOLD}%-30s${RESET}  %s ${CYAN}%3d${RESET}/%d epochs  loss=%-8s acc=%s\n" \
        "$model" "$bar" "$done_epochs" "$TOTAL_EPOCHS" "$train_loss_str" "$train_acc_str"
    printf "  ${DIM}updated: %s${RESET}\n" "$mod_time"

    print_metric_row "val"  "$val_acc"  "$val_clean_acc"  "$val_attack_acc"
    print_metric_row "test" "$test_acc" "$test_clean_acc" "$test_attack_acc"
    echo ""
done
