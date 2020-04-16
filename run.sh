# Adversarial Training (PGD 7)
python train.py --mode 0 --att 1 --train_mode 2 --test_mode 0 --train_max_iter 7 --load_adv_dir "model_chkpt_new/chkpt_plain/" --save_dir "model_chkpt_new/chkpt_pgd7/" --gpu "3,4"

python train.py --mode 0 --att 1 --train_mode 2 --test_mode 0 --train_max_iter 1 --load_adv_dir "model_chkpt_new/chkpt_plain/" --save_dir "model_chkpt_new/chkpt_pgd1/" --gpu "6,7"

