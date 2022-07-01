# Disentanglement multi-task

This repository is a copy of the [disentanglement repository](https://github.com/amir-abdi/disentanglement-pytorch) in pytorch,
modified to address the problem of disentanglement in multi-task learning described in ["On the relationship between disentanglement and multi-task learning"](https://arxiv.org/abs/2110.03498)

### Setting up the project

First, setup the working environment ([see requirements.txt and the instructions for starting](
 https://github.com/amir-abdi/disentanglement-pytorch/tree/master)), and then run:
 
 ```
./train_environ.sh
```

This will set up some constants (alternatively you can set them by hand). 

**NOTE**: This code requires the installation of disentanglement lib. This should be done by
installing the [package](https://pypi.org/project/disentanglement-lib/):

```
pip install disentanglement-lib
```

since installing the disentanglement lib directly from the github source does not seed to work as intended.

### Example Run:

To run the dsprites multitask experiment (sample a task, make a gradient step, both classification and 
regression):

```
./scripts/dsprite_multitask.sh <path_to_folder_where_to_save_outputs>
```

#### Explaining the .sh file: 

Exemplary script looks like this:

```
#!/bin/bash

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME=${1:-$FILENAME}

echo "name=$NAME"

python3 main_mulittask.py \
--name=$NAME \
--alg=DspritesMT \
--dset_dir=./data/test_dsets \
--dset_name=dsprites_full \
--encoder=SimpleConv64 \
--decoder=SimpleConv64 \
--z_dim=8 \
--w_recon=10000 \
--max_iter 250000000 \
--max_epoch 500 \
--header_dim 64 \
--include_labels 0 1 2 3 4 5 \
--train_output_dir "$1/train" \
--test_output_dir "$1/test" \
--ckpt_dir "$1/ckpt_dir" \
--multitask true \
--use_wandb=false \
--batch_size 256 \
--evaluation_metric irs factor_vae_metric mig \
```

Where the first argument is the path to the output folder. In order to change the representation learning
 part of the classifier (encoder, shared parameters between tasks), changed the
 `--encoder` entry. For possible encoders see `architectures/encoders/__init__.py`. 

**NOTE** All encoders with "Gaussian" 
 in the name are probabilistic, and the code is not suited to work with them yet. The choices as for now are:
 `DeepLinear` (for linear) and `SimpleConv64` for convolutional. When writing new encoders or introducing changes please
create a new class and add it to `__init__.py`.
 
Remember to set up the `--dset_dir`  and `--dset_name` properly (for dataset directory and dataset name).

The `--evaluation_metric` allows for specifying the metrics used to evaluate the results. 

**NOTE** They require ground truth data, so will work on dsprites but not CelebA.

The `--include_labels` options allows to specify the labels included into the training. **DO NOT** change this, without changing the number of heads
used by the classifiers. **NOTE**: For CelebA this option has no effect. In order to change the labels_id for CelebA you have to change the 
`CELEB_NUM_CLASSES` and  `CELEB_CLASSES` constants in `common/constants`. 

The training continues until the maximum number of iterations (`--max_iter`) or maximum number of epoch (`--max_epoch`) is reached, so
set either of them (using epochs is preferred).  

In order to see more options and their descriptions, run:

```
python3 main_mulittask.py --help
```
 
For the CelebA  dataset, use: 

```
./scripts/celebA_multitask.sh
```

### Overall Project Structure

#### Encoders

The encoders (shared part of the classical multi-task learning) are specified in 

```
architectures/encoders
```

When creating new encoders, add them to `__init__.py` and make them separate classes derived from
`BaseImageEncoder` (when using images). **DO NOT** overwrite the existing ones. 

#### Headers

The predictor heads are defined in:

```
architectures/headers
```

In particular, for now there is only one-hidden-layer mlp header. If you would like to make other headers than either write 
the new one as a separate class, or create a new base class and make the `Header` class and the new class inherit from it. 
For now the header is loaded directly from the specified file.


#### Models

The models are defined in 

```
models/base_header_model.py 
```                         

and:

```
   models/header_models.py
```

The `BaseHeaderModel` is the base class for all prediction models (classifiers and regressors, `multi-` or `not multi-` task).
Any other models are derived from it. If it is not directly required it is better not to change this class. 

In `header_models.py` there are as for now, four models. 
* `Classifier` (single task), 
* `Regressor` (single task),
* `DspritesMT` (multi-task model),
* `CelebAMT` (multi-task model). 

**NOTE** The dSprites and CelebA multi-task models are hardcoded to have 5 and 10 heads, respectively. 
Ideally, both models should be written in such a manner that they do not require the knowledge about the underling encoder and header type, so this
is the only rule to keep in mind. The headers are still loaded directly from the headers file.

### Datasets and constants

The dataloader, constants and main file arguments are defined in: 
* `common/data_loader.py` - for dataloaders,
* `common/constants.py` - for constants,
* `common/arguments.py` - for command line arguments (an ArgumentParser is used to parse them).

### Metrics

The metrics are specified in the `aicrowd` directory and it is better not to change the files out in there, unless you now what you are doing, and it will work for other users as well. 
Metrics used in paper comes from original `disentanglement-lib` and `aicrowd` repository.

### License
MIT
