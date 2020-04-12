# Adversarial Training (PGD 7)
python train.py --mode 0 --att 1 --train_mode 1 --test_mode 0 --train_max_iter 7 --load_adv_dir "model_chkpt_new/chkpt_plain/" --save_dir "model_chkpt_new/chkpt_pgd7/" --gpu "1,3"

python train.py --mode 0 --att 1 --train_mode 1 --test_mode 0 --train_max_iter 1 --load_adv_dir "model_chkpt_new/chkpt_plain/" --save_dir "model_chkpt_new/chkpt_pgd1/" --gpu "1,3"


# Test Adversarial Models
# (PGD7 : FGSM)
python train.py --mode 1 --att 1 --train_mode 0 --test_mode 1 --train_max_iter 0 --test_max_iter 1 --load_dir "model_chkpt_new/chkpt_pgd7/" --load_adv_dir "model_chkpt_new/chkpt_plain/" --gpu "1,3"

# (PGD7 : PGD20)
python train.py --mode 1 --att 1 --train_mode 0 --test_mode 1 --train_max_iter 0 --test_max_iter 20 --load_dir "model_chkpt_new/chkpt_pgd7/" --load_adv_dir "model_chkpt_new/chkpt_plain/" --gpu "1,3"

# (PGD7 : PGD100)
python train.py --mode 1 --att 1 --train_mode 0 --test_mode 1 --train_max_iter 0 --test_max_iter 100 --load_dir "model_chkpt_new/chkpt_pgd7/" --load_adv_dir "model_chkpt_new/chkpt_plain/" --gpu "1,3"



# (FGSM : FGSM)
python train.py --mode 1 --att 1 --train_mode 0 --test_mode 1 --train_max_iter 0 --test_max_iter 1 --load_dir "model_chkpt_new/chkpt_pgd1/" --load_adv_dir "model_chkpt_new/chkpt_plain/" --gpu "1,3"

# (FGSM : PGD20)
python train.py --mode 1 --att 1 --train_mode 0 --test_mode 1 --train_max_iter 0 --test_max_iter 20 --load_dir "model_chkpt_new/chkpt_pgd1/" --load_adv_dir "model_chkpt_new/chkpt_plain/" --gpu "1,3"

# (FGSM : PGD100)
python train.py --mode 1 --att 1 --train_mode 0 --test_mode 1 --train_max_iter 0 --test_max_iter 100 --load_dir "model_chkpt_new/chkpt_pgd1/" --load_adv_dir "model_chkpt_new/chkpt_plain/" --gpu "1,3"

