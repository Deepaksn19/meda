#!/usr/bin/env bash
# Reproduce the experiments of Elfar et al., IEEE TCAD 2023, end to end.
#
#   bash scripts/reproduce_paper.sh            # full runs (GPU strongly recommended)
#   QUICK=1 bash scripts/reproduce_paper.sh    # smoke-size runs to check the pipeline
#
# Budget: the Table I CNN needs ~1 s per PPO minibatch step on a 4-core CPU,
# so one 2^14-step epoch takes ~35 min there, but minutes on a GPU. The paper
# trains 25-40 epochs per configuration.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-runs}
RES=${RES:-$OUT/figures}      # figures and tables next to the runs
TRIALS=${TRIALS:-1000}
JOBS=${JOBS:-500}
REPEATS=${REPEATS:-5}          # the paper repeats every training experiment 5 times
EXTRA=(--set repeats=$REPEATS)
SIZES=(030 060 090 120 150 180)
TL_ONLY=(); TRAD_ONLY=()   # expanded as ${A[@]+"${A[@]}"}: bash < 4.4 rejects empty "${A[@]}" under set -u
if [[ "${QUICK:-0}" == "1" ]]; then
  EXTRA=(--set repeats=1 --set schedule.epochs=2 --set schedule.steps_per_epoch=2048 --set eval.episodes=50
         --set "agent.extractor_kwargs={channels: [16, 32, 32], hidden_dim: 64}")
  TRIALS=10; JOBS=50; SIZES=(030 060)
  TL_ONLY=(--only s030_f00 s060_f00 s030_f10 s030_f20)
  TRAD_ONLY=(--only s030_f00 s060_f00)
fi
mkdir -p "$RES"

echo "== Figs. 4 (blue), 8: transfer learning chain, rooted at H(30,0%) (stage s030_f00)"
meda curriculum -c configs/curricula/transfer_learning.yaml -o "$OUT" ${TL_ONLY[@]+"${TL_ONLY[@]}"} "${EXTRA[@]}"

echo "== Figs. 4 (red), 7: traditional learning (random init, native resolution)"
meda curriculum -c configs/curricula/traditional_learning.yaml -o "$OUT" ${TRAD_ONLY[@]+"${TRAD_ONLY[@]}"} "${EXTRA[@]}"

for s in "${SIZES[@]}"; do
  meda plot-training "$OUT/traditional_learning/s${s}_f00" "$OUT/transfer_learning/s${s}_f00" \
    --labels "random init" "transfer learning" --title "${s}x${s}" --out "$RES/fig4_${s}.png"
done
for s in s030_f10 s030_f20 s090_f10 s090_f20; do
  [[ -d "$OUT/transfer_learning/$s" ]] || continue
  meda plot-training "$OUT/transfer_learning/$s" --title "$s" --out "$RES/fig8_${s}.png"
done

echo "== Fig. 9: COVID bioassays on a 60x30 chip (one agent, transferred from H*(30,0%))"
meda train -c configs/training/covid_60x30.yaml -o "$OUT" \
  --set init_from="$OUT/transfer_learning/s030_f00/seed_0" "${EXTRA[@]}" --set repeats=1
for assay in covid-rat covid-pcr; do
  # degradation of the authors' Fig. 9 runs (fixed tau = 0.7, c = 200)
  meda bioassay --assay "$assay" --model "$OUT/covid_60x30/seed_0" \
    --routers baseline formal drl --trials "$TRIALS" --out "$RES/fig9" \
    --tau-range 0.7 0.7 --c-range 200 200
done

echo "== Sec. VI: robustness to degraded electrodes (simulation analogue)"
for f in 0.0 0.1; do
  meda compare --model "$OUT/transfer_learning/s030_f10/seed_0" --routers drl baseline formal \
    --jobs "$JOBS" --k-max 40 --set env.fault_fraction=$f --set env.hidden_defect_fraction=0.05 \
    --out "$RES/sec6_faults${f}.csv"
done
echo "done: figures and tables in $RES/"
