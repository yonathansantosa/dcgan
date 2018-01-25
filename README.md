# Deep Convolutional Generative Adversarial Network
This code is part of exercise of Deep Learning for Visual Recognition class. This is the implementation of [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)

The implementation is taken from this [dcgan pytorch example](https://github.com/pytorch/examples)

## How to use

Run `python dcgan.py` to start training

### Arguments :
```
--dataset = 'dataset_name' # load from one of this {lsun, mnist, pokemon, imagenet}
--load          # to load previously trained model
--nonstop       # to train nonstop, without this parameter it will stop at 07.55 PM
--sample=0.3    # fraction of dataset that will be used in training. 1 for all data
--save=100      # iteration before saving each image
```