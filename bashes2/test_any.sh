#! bin/bash
DEVICE=0
TASK_str=stack_block
MODEL=1Task-NUTASSEMBLY-Batch9-1gpu-Attn2ly128-Act2ly64mix2-headCat-simclr128x256

python3 ../tasks/test_models/test_one_model.py $MODEL --last_few 1 --eval_tasks ${TASK_str} --num_workers 3