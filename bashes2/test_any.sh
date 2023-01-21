#! bin/bash
DEVICE=0
TASK_str=nut_assembly
MODEL=/home/ciccio/Desktop/multi_task_lfd/Multi-Task-LFD-Framework/training_log/1Task-NUT-ASSEMBLY-Batch5-1gpu-Attn2ly128-Act2ly64mix2-headCat-simclr128x256

python3 ../tasks/test_models/test_one_model.py $MODEL --last_few 1 --eval_tasks ${TASK_str} --num_workers 1