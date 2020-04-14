# Adversarial Training in PyTorch
This is an implementation of adversarial training using the 
***Fast Gradient Sign Method*** (FGSM) [[1]](https://arxiv.org/abs/1412.6572),
***Projected Gradient Descent*** (PGD) [[2]](https://arxiv.org/abs/1706.06083), and 
***Momentum Iterative FGSM*** (MI-FGSM) [[3]](https://arxiv.org/abs/1710.06081)
attacks to generate adversarial examples. The model employed to compute
adversarial examples is WideResNet-28-10 [[4]](https://arxiv.org/abs/1605.07146).
An implementation of this model is retrieved from [[5]](https://github.com/xternalz/WideResNet-pytorch).
The dataset used to conduct the experiment is CIFAR-10.

## Usage
### Installation
The training environment (PyTorch and dependencies) can be installed as follows:
```
git clone https://github.com/AlbertMillan/adversarial-training-pytorch.git
python setup.py install
```

Tested under Python 3.8.0 and PyTorch 1.4.0

### Arguments
This model offers a significant degree of customization. The following are the list of arguments:
##### Storage Variables

| Command          | Default Value               | Description   |
| ---------------- |:---------------------------:|:------------- |
| --ds_path        | 'datasets/'                 | Path to dataset.
| --load_dir       | 'chkpt/chkpt_plain/'        | Path to pre-trained model. Used to generate adversarial examples from the test set.
| --load_name      | 'chkpt__model_best.pth.tar' | 
| --load_adv_dir   | 'chkpt/chkpt_plain/'        |
| --load_adv_name  | 'chkpt__model_best.pth.tar' | File name
| --save_dir       | 'chkpt/new/'                | Path to store model checkpoints on each iteration.

##### Model Hyper-parameters

| Command        | Default Value | Description                             |
|----------------|:-------------:|:--------------------------------------- |
| --lr           | 0.1           | Learning rate.
| --itr          | 76            | Number of training iterations.
| --batch_size   | 64            | Batch size.
| --momentum     | 0.9           | Momentum constant.
| --nesterov     | True          | Whether to apply Nesterov momentum.
| --weight_decay | 2e-4          | Weight decay.
| --topk         | 1             | Compute accuracy over top k-predictions

##### Adversarial Generator Properties
| Command          | Default Value  | Description   |
| ---------------- |:--------------:|:------------- |
| --eps            | (8./255.)      | Epsilon (float)
| --attack         | 0              | Attack type (0: no-attack; 1: PGD)
| --adv_momentum   | None           | Momentum constant used to generate adversarial examples if given (float).
| --train_max_iter | 1              | Iterations performed to generate adversarial examples from train set.
| --test_max_iter  | 0              | Iterations performed to generate adversarial examples from test set.
| --train_mode     | 0              | Training on raw images (0), adversarial images (1) or both (2).
| --test_mode      | 0              | Testing on raw images (0), adversarial images (1) or both (2).

##### Other Properties
| Command          | Default Value  | Description   |
| ---------------- |:--------------:|:------------- |
| --gpu            | "0,1"          | Epsilon
| --zero_norm      | False          | Whether to perform zero-mean normalization on the dataset.
| --skip_train     | False          | Wether to perform testing without training, loading pre-trained model.





## Setup

## Examples of Use

## Acknowledgements
