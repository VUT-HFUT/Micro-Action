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

```bash
python tools/train.py configs/recognition/manet/manet.py --seed=0 --deterministic

```

## Evaluation 
We provide the pre-trained weights of MANet, you can download it from [here](https://huggingface.co/kunli-cs/MANet_weights/resolve/main/MANet/best_top1_acc_epoch_40.pth?download=true)

``` bash
python tools/test.py configs/recognition/manet/manet.py work_dirs/manet/best_top1_acc_epoch_40.pth --out online_evaluation/test_result.pickle
python online_evaluation/eval.py
```

### Codabench Submission (Test set)

### Format
The test set is reserved for competition. 
The prediction file `prediction.csv` is in csv format, each row of the files denotes the predicted categories. 
```
          `id`: `video id`
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