#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate l3mvn

HEIGHTS=(0.75 1.0 1.25 1.5)
FOVS=(80 95 110 125)

total_scenes=80 # train scenes
slices=$(( ${#HEIGHTS[@]} * ${#FOVS[@]} ))
n=$(( total_scenes / slices ))

# max processes on single 3090/4090
if [ "$n" -gt 6 ]; then
    n=6
elif [ "$n" -lt 1 ]; then
    n=1
fi

echo "Total slices: $slices"
echo "Number of processes: $n"

run_id=0

for h in "${HEIGHTS[@]}"; do
    for f in "${FOVS[@]}"; do
        echo "Run $run_id: height=$h, fov=$f"

        python main_llm_vis.py \
            --split train \
            --eval 1 \
            --auto_gpu_config 0 \
            --load pretrained_models/llm_model.pt \
            --use_gtsem 0 \
            --num_local_steps 10 \
            -n $n \
            --num_eval_episodes 50 \
            --camera_height "$h" \
            --hfov "$f" \
            --scene_slice_total $slices \
            --scene_slice_index "$run_id"

        run_id=$((run_id + 1))
    done
done
