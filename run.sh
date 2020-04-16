# Plain Training
# python train.py --mode 0 --att 0 --train_mode 0 --test_mode 0 --zero_norm 0 --save_dir "model_chkpt_new/chkpt_scaled/" --gpu "0,1,2,3" &

# python train.py --mode 0 --att 0 --train_mode 0 --test_mode 0 --zero_norm 1 --save_dir "model_chkpt_new/chkpt_norm/" --gpu "4,5,6,7"

# python train.py --mode 0 --att 1 --train_mode 2 --test_mode 0 --train_max_iter 7 --load_adv_dir "model_chkpt_new/chkpt_plain/" --save_dir "model_chkpt_new/chkpt_pgd7/" --gpu "1,3"


# Adversarial Training
# (NORM :: NO_RETAIN :: FGSM)
python train.py --mode 0 --att 1 --train_mode 2 --test_mode 0 --train_max_iter 1 --zero_norm 1 --store_adv 0 --load_adv_dir "model_chkpt_new/chkpt_norm/" --save_dir "model_chkpt_new/chkpt__norm__noretain__pgd1/" --gpu "0,1" &

# # (NORM :: NO_RETAIN :: PGD7)
python train.py --mode 0 --att 1 --train_mode 2 --test_mode 0 --train_max_iter 7 --zero_norm 1 --store_adv 0 --load_adv_dir "model_chkpt_new/chkpt_norm/" --save_dir "model_chkpt_new/chkpt__norm__noretain__pgd7/" --gpu "2,3" & 

# # (NORM :: RETAIN :: FGSM)
python train.py --mode 0 --att 1 --train_mode 2 --test_mode 0 --train_max_iter 1 --zero_norm 1 --store_adv 1 --load_adv_dir "model_chkpt_new/chkpt_norm/" --save_dir "model_chkpt_new/chkpt__norm__retain__pgd1/" --gpu "4,5" &

# # (NORM :: RETAIN :: PGD7)
python train.py --mode 0 --att 1 --train_mode 2 --test_mode 0 --train_max_iter 7 --zero_norm 1 --store_adv 1 --load_adv_dir "model_chkpt_new/chkpt_norm/" --save_dir "model_chkpt_new/chkpt__norm__retain__pgd7/" --gpu "6,7" &



# (NORM :: NO_RETAIN :: FGSM)
# python train.py --mode 0 --att 1 --train_mode 2 --test_mode 0 --train_max_iter 1 --store_adv 0 --load_adv_dir "model_chkpt_new/chkpt_scaled/" --save_dir "model_chkpt_new/chkpt__scaled__noretain__pgd1/" --gpu "0,1" &

# (NORM :: NO_RETAIN :: PGD7)
python train.py --mode 0 --att 1 --train_mode 2 --test_mode 0 --train_max_iter 7 --store_adv 0 --load_adv_dir "model_chkpt_new/chkpt_scaled/" --save_dir "model_chkpt_new/chkpt__scaled__noretain__pgd7/" --gpu "2,3"

# (NORM :: RETAIN :: FGSM)
# python train.py --mode 0 --att 1 --train_mode 2 --test_mode 0 --train_max_iter 1 --store_adv 1 --load_adv_dir "model_chkpt_new/chkpt_scaled/" --save_dir "model_chkpt_new/chkpt__scaled__retain__pgd1/" --gpu "4,5" &

# (NORM :: RETAIN :: PGD7)
# python train.py --mode 0 --att 1 --train_mode 2 --test_mode 0 --train_max_iter 7 --store_adv 1 --load_adv_dir "model_chkpt_new/chkpt_scaled/" --save_dir "model_chkpt_new/chkpt__scaled__retain__pgd7/" --gpu "6,7"  




