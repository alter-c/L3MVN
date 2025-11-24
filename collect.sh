#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate l3mvn

FOVS=(75 90 105 120)
HEIGHTS=(0.6 0.9 1.2 1.5)

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
            -n 5 \
            --num_eval_episodes 100 \
            --camera_height "$h" \
            --hfov "$f" \
            --scene_slice_total 16 \
            --scene_slice_index "$run_id"

        run_id=$((run_id + 1))
    done
done
