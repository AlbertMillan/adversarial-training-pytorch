# Test Adversarial Models
# (STANDART : FGSM)
# python train.py --mode 1 --att 1 --train_mode 0 --test_mode 1 --train_max_iter 0 --test_max_iter 1 --load_dir "model_chkpt_new/chkpt_plain/" --load_adv_dir "model_chkpt_new/chkpt_plain/" --save_idx 1 --gpu "0,5" &

# (STANDART : PGD20)
# python train.py --mode 1 --att 1 --train_mode 0 --test_mode 1 --train_max_iter 0 --test_max_iter 20 --load_dir "model_chkpt_new/chkpt_plain/" --load_adv_dir "model_chkpt_new/chkpt_plain/" --save_idx 2 --gpu "1,2" &

# (STANDART : PGD100)
# python train.py --mode 1 --att 1 --train_mode 0 --test_mode 1 --train_max_iter 0 --test_max_iter 100 --load_dir "model_chkpt_new/chkpt_plain/" --load_adv_dir "model_chkpt_new/chkpt_plain/" --save_idx 3 --gpu "1,2" &

# (STANDART : MIM)
# python train.py --mode 1 --att 1 --train_mode 0 --test_mode 1 --train_max_iter 0 --test_max_iter 20 --adv_momentum 1.0 --load_dir "model_chkpt_new/chkpt_plain/" --load_adv_dir "model_chkpt_new/chkpt_plain/" --save_idx 4 --gpu "1,2"




# (FGSM : FGSM)
# python train.py --mode 1 --att 1 --train_mode 0 --test_mode 1 --train_max_iter 0 --test_max_iter 1 --load_dir "model_chkpt_new/chkpt_pgd1/" --load_adv_dir "model_chkpt_new/chkpt_plain/" --save_idx 9 --gpu "3,4" &

# (FGSM : PGD20)
# python train.py --mode 1 --att 1 --train_mode 0 --test_mode 1 --train_max_iter 0 --test_max_iter 20 --load_dir "model_chkpt_new/chkpt_pgd1/" --load_adv_dir "model_chkpt_new/chkpt_plain/" --save_idx 10 --gpu "3,4" &

# (FGSM : MIM)
# python train.py --mode 1 --att 1 --train_mode 0 --test_mode 1 --train_max_iter 0 --test_max_iter 20 --adv_momentum 1.0 --load_dir "model_chkpt_new/chkpt_pgd1/" --load_adv_dir "model_chkpt_new/chkpt_plain/" --save_idx 12 --gpu "6,7"

# (FGSM : PGD100)
# python train.py --mode 1 --att 1 --train_mode 0 --test_mode 1 --train_max_iter 0 --test_max_iter 100 --load_dir "model_chkpt_new/chkpt_pgd1/" --load_adv_dir "model_chkpt_new/chkpt_plain/" --save_idx 11 --gpu "3,4" &





# (PGD7 : FGSM)
# python train.py --mode 1 --att 1 --train_mode 0 --test_mode 1 --train_max_iter 0 --test_max_iter 1 --load_dir "model_chkpt_new/chkpt_pgd7/" --load_adv_dir "model_chkpt_new/chkpt_plain/" --save_idx 5 --gpu "3,4" &

# (PGD7 : PGD20)
# python train.py --mode 1 --att 1 --train_mode 0 --test_mode 1 --train_max_iter 0 --test_max_iter 20 --load_dir "model_chkpt_new/chkpt_pgd7/" --load_adv_dir "model_chkpt_new/chkpt_plain/" --save_idx 6 --gpu "6,7" & 

# (PGD7 : PGD100)
python train.py --mode 1 --att 1 --train_mode 0 --test_mode 1 --train_max_iter 0 --test_max_iter 100 --load_dir "model_chkpt_new/chkpt_pgd7/" --load_adv_dir "model_chkpt_new/chkpt_plain/" --save_idx 7 --gpu "3,4"

# (PGD7 : MIM)
# python train.py --mode 1 --att 1 --train_mode 0 --test_mode 1 --train_max_iter 0 --test_max_iter 20 --adv_momentum 1.0 --load_dir "model_chkpt_new/chkpt_pgd7/" --load_adv_dir "model_chkpt_new/chkpt_plain/" --save_idx 8 --gpu "1,2"