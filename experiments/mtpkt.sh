cd ..
python fmbo_main.py --num_runs 20 --iid --total_clients 10 --multi_task --dim 10 --knowledge_transfer --save_results
python fmbo_main.py --num_runs 20 --iid --total_clients 10 --multi_task --acq_type EI_w --dim 10 --knowledge_transfer --save_results