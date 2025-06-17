#

## Data preparation
Download the dataset from [Huggingface](https://huggingface.co/datasets/kunli-cs/MA-52/tree/main). 

Put dataset into `./data/ma52`.

## Installation
```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch torchvision -c pytorch  # This command will automatically install the latest version PyTorch and cudatoolkit, please check whether they match your environment.
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet  # optional
mim install mmpose  # optional
git clone https://github.com/VUT-HFUT/Micro-Action.git
cd Micro-Action/mar_scripts/manet/mmaction2
pip install -v -e .
```
Recommeded Version: 
```
torch==1.12.1
torchvision==0.13.1
mmaction2==0.24.1
mmcv-full==1.6.2
```

## Training

### MANet
```bash
python tools/train.py configs/recognition/manet/manet.py --seed=0 --deterministic
```

## Evaluation 
We provide the pre-trained weights of MANet, you can download it from [here](https://huggingface.co/kunli-cs/MANet_weights/resolve/main/MANet/best_top1_acc_epoch_40.pth?download=true)

``` bash
## for MANet
python tools/test.py configs/recognition/manet/manet.py work_dirs/manet/best_top1_acc_epoch_40.pth --out online_evaluation/test_result.pickle
python online_evaluation/eval.py
```

### Codabench Submission (Test set)

### Format
The test set is reserved for competition. 
The prediction file `prediction.csv` is in csv format, each row of the files denotes the predicted categories. 
```
          `vid`: `video id`
`action_pred_1`: `top 1 action-level category`
`action_pred_2`: `top 2 action-level category`
`action_pred_3`: `top 3 action-level category`
`action_pred_4`: `top 4 action-level category`
`action_pred_5`: `top 5 action-level category`
`body_pred_1`: `top 1 body-level category`
`body_pred_2`: `top 2 body-level category`
`body_pred_3`: `top 3 body-level category`
`body_pred_4`: `top 4 body-level category`
`body_pred_5`: `top 5 body-level category`
```

To test your model's performance on `test` split, please submit the test predictions to the [Codabench evaluation server](https://www.codabench.org/competitions/9066/). The submission file should be a single `.zip` file (no enclosing folder) that contains the prediction file `prediction.csv` formmated as instructed above.  


## Results
| Method | Body Top-1 | Action Top-1 | Action Top-5 | Body F1_Macro | Body F1 Micro | Action F1 Macro | Action F1 Macro | F1 mean | Config	| Download |
| ------ | ------ | ------ | ------ | ------ | ------ | ------- | ------- | ------- | ------- | ------- | 
| C3D | 74.04 |  52.22 |  86.97 |  66.60 |  74.04 | 40.86 | 52.22 | 58.43 | - | - |
| I3D | 78.16 |  57.07 |  88.67 |  71.56 |  78.16 | 39.84 | 57.07 | 61.66 | - | - |
| SlowFast | 77.18 | 59.60 | 88.54 | 70.61 | 77.18 | 44.96 | 59.60 | 63.09 | - | - |
| TSM | 77.64 |  56.75 |  87.47 |  70.98 |  77.64 | 40.19 | 56.75 | 61.39 | - | - |
| MANet | 78.95 | 61.33 | 88.83 | 72.87 | 78.95 | 49.22 | 61.33 | **65.59** | [config](manet/mmaction2/configs/recognition/manet/manet.py) | [model](https://huggingface.co/kunli-cs/MANet_weights/resolve/main/MANet/best_top1_acc_epoch_40.pth?download=true) |

## Acknoledgements 
This code began with [mmaction2](https://github.com/open-mmlab/mmaction2). We thank the developers for doing most of the heavy-lifting. 


## Citation 
Please consider citing the related paper in your publications if it helps your research.

```
@article{guo2024benchmarking,
  title={Benchmarking Micro-action Recognition: Dataset, Methods, and Applications},
  author={Guo, Dan and Li, Kun and Hu, Bin and Zhang, Yan and Wang, Meng},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024},
  volume={34},
  number={7},
  pages={6238-6252}
}
```
