#!/bin/bash

cd /bd_byta6000i0/users/jhu/DEX/
source /bd_targaryen/users/kleelakunwet/miniconda3/bin/activate
conda activate dex

obstacles=(
    # [1]="Sphere" 
    # [2]="Surface" 
    # [3]="Plate"
    # [4]="HalfSphere"
    # [5]="Cyclinder"
    [6]="ComplexCyclinder"
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
        python eval.py task=$task n_eval_episodes=50 render_three_views=True
        # python video.py ${task%-*}-$obstacle

        python eval.py task=$task n_eval_episodes=50 use_dcbf=True render_three_views=True
        # python video.py ${task%-*}-$obstacle-CBF

        python eval.py task=$task n_eval_episodes=50 use_dcbf=True use_dclf=True render_three_views=True
        # python video.py ${task%-*}-$obstacle-CBF-CLF
    done
done
