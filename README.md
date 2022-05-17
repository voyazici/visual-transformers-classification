# visual-transformers-classification

This repository provides the MS-COCO training code for the [Visual Transformers with Primal Object Queries for Multi-Label Image Classification](https://arxiv.org/abs/2112.05485) paper which will be published in ICPR2022.

## Environment
* python 3.7
* pytorch 1.6.0

## Training
python train.py -image_path <image_path> -save_path <save_path> -mix_up

## Testing
python train.py -image_path <image_path> -snapshot <snapshot> -test_model

## Acknowledgements
Transformer encoder-decoder models in this repository are based on the implementation in [here](https://github.com/facebookresearch/detr).
