#!/bin/sh
. $PREFIX/etc/profile.d/conda.sh  # do not edit
conda activate $PREFIX            # do not edit

# add your pip packages here; bottle is just an example!
# replace it with your dependencies
python -m pip install matplotlib==3.5.1 mapcalc==0.2.2 pytorch-lightning==1.4.9 torch-cluster==1.6.1 torch-geometric==2.0.4 torch-scatter==2.1.1 torch-sparse==0.6.13 torch-spline-conv==1.2.2 torchmetrics==0.4.0 wandb==0.14.0