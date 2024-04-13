# Micro-Action Recognition Benchmark

**Benchmarking Micro-action Recognition: Dataset, Methods, and Applications**

[[Paper](https://ieeexplore.ieee.org/document/10414076)] 


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/benchmarking-micro-action-recognition-dataset/micro-action-recognition-on-ma-52)](https://paperswithcode.com/sota/micro-action-recognition-on-ma-52?p=benchmarking-micro-action-recognition-dataset)


## News




## Introduction
Micro-action is an imperceptible non-verbal behaviour characterised by low-intensity movement. It offers insights into the feelings and intentions of individuals and is important for human-oriented applications such as emotion recognition and psychological assessment. However, the identification, differentiation, and understanding of micro-actions pose challenges due to the imperceptible and inaccessible nature of these subtle human behaviors in everyday life. In this study, we innovatively collect a new micro-action dataset designated as Micro-action-52 (MA-52), and propose a benchmark named micro-action network (MANet) for micro-action recognition (MAR) task. Uniquely, MA-52 provides the whole-body perspective including gestures, upper- and lower-limb movements, attempting to reveal comprehensive micro-action cues. In detail, MA-52 contains 52 micro-action categories along with seven body part labels, and encompasses a full array of realistic and natural micro-actions, accounting for 205 participants and 22,422 video instances collated from the psychological interviews. Based on the proposed dataset, we assess MANet and other nine prevalent action recognition methods. MANet incorporates squeeze-and-excitation (SE) and temporal shift module (TSM) into the ResNet architecture for modeling the spatiotemporal characteristics of micro-actions. Then a joint-embedding loss is designed for semantic matching between video and action labels; the loss is used to better distinguish between visually similar yet distinct micro-action categories. The extended application in emotion recognition has demonstrated one of the important values of our proposed dataset and method. In the future, further exploration of human behaviour, emotion, and psychological assessment will be conducted in depth. 

## Data

### Micro-Action examples


## Code
We will release the source code soon. 

## Download 
The datasets are **only** to be used for **non-commercial scientific purposes**. If you want use the datasets, please feel free to send an email to this [address](guodan@hfut.edu.cn). 


## Citation

Please consider citing the related paper in your publications if it helps your research.

```
@article{guo2024benchmarking,
  title={Benchmarking Micro-action Recognition: Dataset, Methods, and Applications},
  author={Guo, Dan and Li, Kun and Hu, Bin and Zhang, Yan and Wang, Meng},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024},
  publisher={IEEE}
  doi={10.1109/TCSVT.2024.3358415}
}
```