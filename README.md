# Disentanglement-multi-task

This repository is a copy of the disentanglement repository in pytorch 
(https://github.com/amir-abdi/disentanglement-pytorch/tree/master), 
modified to address the problem of disentanglement in multi-task learning.

### Setting Up the Project:

First, setup the working environment (see requirements.txt and the instructions for starting in
 https://github.com/amir-abdi/disentanglement-pytorch/tree/master), and then run:
 
 ```
./train_environ.sh
```

This will set up some constants (alternatively you can set them by hand). 

**NOTE**: This code requires the installation of disentanglement lib. This should be done by:
installing the package (https://pypi.org/project/disentanglement-lib/):

```
pip install disentanglement-lib
```

since installing the disentanglement lib directly from the github source does not seed to work as intended.

**NOTE** I changed the way CelebA dataset is loaded - now it simply uses the torchvision dataset 
(note that the CelebA dataset with the required torchvision file format is available at our servers, 
so DO NOT download it again privately. The path do the dataset is set in `scripts/celebA_multitask.sh`).
In order for this to work, you need a newer version of pytorch than the one specified in requirements.txt (>= 1.3), 
since the CelebA dataset was not previously included in torchvision.


### Example Run:

To run the dsprites multitask experiment (old version - sample a task, make a gradient step, both classification and 
regression):

```angular2
./scripts/dsprite_multitask.sh <path_to_folder_where_to_save_outputs>
```

#### Examplining the .sh file: 

The script looks like this:

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
 part of the classifier (encoder, shared parameters between tasks), change the
 `--encoder` entry. For possible encoders see `architectures/encoders/__init__.py`. All encoders with "Gaussian" 
 in the name are probabilistic, and the code is not suited to work with them yet. So basicly the choices as for now are:
 `DeepLinear` (for linear) and `SimpleConv64` for convolutional. When writing new encoders or introducing changes please
 **DO NOT** overwrite the existing ones. Instead, create a new class and add it to `__init__.py`.
 
Remember to set up the `--dset_dir`  and `--dset_name` properly (I should work with the current configuration, but if you have the
dsprites dataset downloaded elsewhere, then you can change the dataset dir.)

The `--evaluation_metric` allows for specifying the metrics used to evaluate the results. They require ground truth data,
so will work on dsprites but not CelebA.

The `--include_labels` options allows to specify the labels included into the training. DO NOT chanege this, without changing the number of heads
used by the classfiers (For now the number of heads is hardcoded for 5 for dsprites - since label 0 is always 0 - and 10 for 
CelebA). **NOTE**: For CelebA this option has no effect. In order to change the labels_id for CelebA you have to change the 
`CELEB_NUM_CLASSES` and  `CELEB_CLASSES` constants in `common/constants`. 

The training continues until the maximum number of iterations (`max_iter`) or maximum number of epoch (`max_epoch`) is reached, so
set either of them (using epochs is preferred).  

In order to see more options and their descriptions, run:

```
python3 main_mulittask.py --help
```
 
For the CelebA  dataset, use: 

```
./scripts/celebA_multitask.sh
```

# Overall Project Structure

### Architectures

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
For now the header is loaded directly from the specified file (so is not present in the __init__.py, but this will most likely change in the future. )


#### Models

The models are defined in 

```
models/base_header_model.py 
```                         

and:

```
   models/header_models.py
```

The `BaseHeaderModel` is the base class for all prediction models (classifiers and regressors, multi- ot not multi- task).
Any other models are derived from it. If it is not directly required it is better not to change this class. 

In `header_models.py` there are as for now, four models. A `Classifier` (single task), a `Regressor` (single task) 
and a `DspritesMT` model and a `CelebAMT` model. There are still prototypical, so changing them (and overwriting) is allright, as long as you update this
`README` accordingly, if required. For now both the dsprites and CelebA multi-task models are hardcoded to have 5 and 10 heads, respectively. 
Ideally, both models should be written in such a manner that they do not require the knowledge about the underling encoder and header type, so this
is the only rule to keep in mind. The headers are still loaded directly from the headers file, but this will be probably changed to loading a model by it's name, as
specified in `__init__.py` (This is a small cosmetic change, so you don't have to bother about this right now). 

### Datasets and constants

The dataloader, constants and main file arguments are defined in 
`common/data_loader.py` - for dataloaders.
`common/constants.py` - for constants.
`common/arguments.py` - for command line arguments. An ArgumentParser is used to parse them. 

### Metrics

The metrics are specified in the `aicrowd` directory and it is better not to change the files out in there, unless you now what you are doing, and it will work for other users as well. 
I guess the best way to add new metrics would be to create a new file in this directory that contains them, so we could keep the ones we
implemented ourselves separate from the ones from disentanglement-lib and aicrowd.

### Main File

The main file is the `main_multitask.py` file. 

# Github Branches

It is fine (and advised) to work with different branches. Just remember to update them frequently with the master, to be sure that we work
more or less with the same code. Otherwise merging can become a mess. If you are introducing any significant changes that we did not discussed on the meetings,
please update this README, so we will all be aware of them.



