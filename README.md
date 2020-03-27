# Overview
<img src="https://github.com/dmt-lab/learning-to-observe/blob/master/github_image.png" width="400px">

This repository contains the code and models associated with our CVPR publication titled: *"Learning to Observe: Approximating Human Perceptual Thresholds for Detection of Suprathreshold Image Transformations"*

**Learning to Observe: Approximating Human Perceptual Thresholds for Detection of Suprathreshold Image Transformations** <BR>
Alan Dolhasz, Carlo Harvey, Ian Williams <BR>
in CVPR 2020 <BR>

```
@article{dolhasz2020learning,
  title={Learning to Observe: Approximating Human Perceptual Thresholds for Detection of Suprathreshold Image Transformations},
  author={Dolhasz, Alan and Harvey, Carlo and Williams, Ian},
  journal={CVPR},
  year={2020}
}
```

Preprint: https://arxiv.org/pdf/1912.06433.pdf

Please refer to the paper for more details. Instructions on how to download and use our pretrained models can be found below.

# Installation & Setup

1. Setup a conda environment:
```
conda env create -f ./cvpr_env.yml --name cvpr
```

2. Activate environment:
```
conda activate cvpr
```

3. Install Cython
```
pip install cython
```

4. Install remaining pip requirements:
```
pip install -r ./requirements.txt
```
<br>

# Download pre-trained models
```
python ./download_models.py
```
This script will download a pretrained model trained on `./data/dataset_fold_5.csv` with the enitre encoder frozen (highest-performing freeze point).
If you would like to download models trained on folds 1-4 as well, please use the `--all_folds` option.

<br>

# (Optional) Change configs
To evaluate models trained on folds other than fold 5, please make sure to change this in the `./configs/classifier_config.py` file

# Predict and Visualise results
(Uses fold 5 by default, if no config changes are made)

```
python predict.py --model ./data/pretrained_classifier_model.hdf5 --dataset val
```

<br>

# AET Training
To train the AET backbone network in an unsupervised manner, please make sure to download the 2017 MSCOCO dataset and update the `COCO_PATH` variable in `./configs/aet_config.py` to the location of your MSCOCO directory. See http://cocodataset.org/#download for more information.

Please also make sure to set `BATCH_SIZE` to fit your GPU.

Then run:
```
python ./train_aet.py
```

<br>

# Perceptual Threshold Classifier training
If training from your own AET backbone, please make sure to change the `AET_PATH` variable to the path of your trained AET. 

Then:
```
python ./train_classifier.py
```
