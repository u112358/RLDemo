#!/usr/bin/env bash
# Controlled comparison: one change per run against the same baseline, same seed, same number of
# gradient updates.  Edit BUDGET / SEED / LAYERS; every run keeps its files in cache/<tag>/.
# Works with the bash 3.2 that ships with macOS (no associative arrays).
#
#   bash run_matrix.sh            # runs all five, then prints the comparison table
#   bash run_matrix.sh base cubie # only these
set -e
BUDGET=${BUDGET:-40000}          # gradient updates per run (M1 Max, 1024-1024-512, batch 4096: ~30-40 min each)
SEED=${SEED:-0}
LAYERS=${LAYERS:-1024,1024,512}
COMMON="--agent net --backend torch --layers $LAYERS --batch 4096 --updates $BUDGET --lr-final 1e-4 --promote 0.9 --seed $SEED --plain"
flags() {
  case "$1" in
    base)    echo "" ;;                       # sticker features, greedy promotion, no augmentation
    onestep) echo "--promote-by onestep" ;;   # curriculum advances on one-step accuracy
    beam)    echo "--promote-by beam" ;;      # curriculum advances on beam-search success (width 8)
    cubie)   echo "--features cubie" ;;       # cubie position + twist input instead of raw stickers
    symaug)  echo "--symmetry-aug" ;;         # every sample replaced by a random symmetric image
    *) echo "unknown run: $1" >&2; exit 1 ;;
  esac
}
ORDER=(base onestep beam cubie symaug)
if [ $# -gt 0 ]; then ORDER=("$@"); fi
for tag in "${ORDER[@]}"; do
  extra=$(flags "$tag")
  echo "=== $tag $extra"
  python3 play_rubik.py train $COMMON --tag "m_$tag" $extra 2>&1 | tee "cache/matrix_$tag.log" | grep -v "^step" || true
done
python3 compare.py $(for t in "${ORDER[@]}"; do echo -n "m_$t "; done)
