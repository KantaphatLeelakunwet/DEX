#!/bin/bash

obstacles=(
    [1]="Sphere" 
    [2]="Surface" 
    [3]="Box"
    [4]="HalfSphere"
)

for key in "${!obstacles[@]}"; do
    tasks=("NeedlePick-v$key" "NeedleReach-v$key" "GauzeRetrieve-v$key" "PegTransfer-v$key")
    obstacle=${obstacles[$key]}
    for task in "${tasks[@]}"; do
        python eval.py task=$task n_eval_episodes=50 &> log/${task%-*}-$obstacle
        python video.py ${task%-*}-$obstacle

        python eval.py task=$task n_eval_episodes=50 use_dcbf=True &> log/${task%-*}-$obstacle-CBF
        python video.py ${task%-*}-$obstacle-CBF

        python eval.py task=$task n_eval_episodes=50 use_dcbf=True use_dclf=True &> log/${task%-*}-$obstacle-CBF-CLF
        python video.py ${task%-*}-$obstacle-CBF-CLF
    done
done
