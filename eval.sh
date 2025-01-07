#!/bin/bash

obstacles=(
    [1]="Sphere" 
    # [2]="Surface" 
    # [3]="Plate"
    # [4]="HalfSphere"
    # [5]="Cylinder"
    # [6]="ComplexCylinder"
)

for key in "${!obstacles[@]}"; do
    tasks=(
        "NeedlePick-v$key" 
        "NeedleReach-v$key" 
        "GauzeRetrieve-v$key" 
        "PegTransfer-v$key"
        # "NeedleRegrasp-v$key"
    )
    obstacle=${obstacles[$key]}
    
    for task in "${tasks[@]}"; do
        python eval.py task=$task n_eval_episodes=50 render_three_views=True &> log/${task%-*}-$obstacle &
        # python video.py ${task%-*}-$obstacle

        python eval.py task=$task n_eval_episodes=50 use_dcbf=True render_three_views=True &> log/${task%-*}-$obstacle-CBF &
        # python video.py ${task%-*}-$obstacle-CBF

        python eval.py task=$task n_eval_episodes=50 use_dcbf=True use_dclf=True render_three_views=True &> log/${task%-*}-$obstacle-CBF-CLF &
        # python video.py ${task%-*}-$obstacle-CBF-CLF
    done
done
