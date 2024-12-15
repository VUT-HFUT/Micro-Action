# Micro-Action Benchmark

**Benchmarking Micro-action Recognition: Dataset, Methods, and Applications**

[[Paper](https://ieeexplore.ieee.org/document/10414076)] 


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/benchmarking-micro-action-recognition-dataset/micro-action-recognition-on-ma-52)](https://paperswithcode.com/sota/micro-action-recognition-on-ma-52?p=benchmarking-micro-action-recognition-dataset)


## News
- [2024/12/15] :tada: :tada: :tada: Our paper [PCAN](https://github.com/kunli-cs/PCAN) about Micro-Action Recognition is accepted by **AAAI 2025**. 
- [2024/7/9] :tada: :tada: :tada: We released the MMA-52 dataset for Multi-label Micro-Action Detection task. [Report](https://arxiv.org/abs/2407.05311)
- [2024/4/26] We launched the Micro-Action Analysis Grand Challenge ([MAC 2024](https://sites.google.com/view/micro-action)) associated with [ACM Multimedia 2024](https://2024.acmmm.org/).
- [2024/4/16] We released the source code of MANet. 

---

## Introduction
Micro-action is an imperceptible non-verbal behaviour characterised by low-intensity movement. It offers insights into the feelings and intentions of individuals and is important for human-oriented applications such as emotion recognition and psychological assessment. However, the identification, differentiation, and understanding of micro-actions pose challenges due to the imperceptible and inaccessible nature of these subtle human behaviors in everyday life. In this study, we innovatively collect a new micro-action dataset designated as Micro-action-52 (MA-52), and propose a benchmark named micro-action network (MANet) for micro-action recognition (MAR) task. Uniquely, MA-52 provides the whole-body perspective including gestures, upper- and lower-limb movements, attempting to reveal comprehensive micro-action cues. In detail, MA-52 contains 52 micro-action categories along with seven body part labels, and encompasses a full array of realistic and natural micro-actions, accounting for 205 participants and 22,422 video instances collated from the psychological interviews. Based on the proposed dataset, we assess MANet and other nine prevalent action recognition methods. MANet incorporates squeeze-and-excitation (SE) and temporal shift module (TSM) into the ResNet architecture for modeling the spatiotemporal characteristics of micro-actions. Then a joint-embedding loss is designed for semantic matching between video and action labels; the loss is used to better distinguish between visually similar yet distinct micro-action categories. The extended application in emotion recognition has demonstrated one of the important values of our proposed dataset and method. In the future, further exploration of human behaviour, emotion, and psychological assessment will be conducted in depth. 

---

## Data

### Download

The datasets are **only** to be used for **non-commercial scientific purposes**. You may request access to the dataset by completing the Google Form provided and corresponding LA files. We will respond promptly upon receipt of your application. If you have difficulty in filling out the form, we can also accept the application by [[email](mailto:kunli.hfut@gmail.com?subject=Micro-Action%20Dataset%20Requests&cc=guodan@hfut.edu.cn)]. 

- Micro-Action-52 dataset (MA-52): [Application Form](https://forms.gle/avQQiRWvbxa1nDFQ6) and [LA File](https://drive.google.com/file/d/1vAussMwE9GrL5Vt1MpSQeSmVbUMsgPhw/view?usp=sharing) 
- Micro-Action-Pro dataset (MA-52-Pro): [Application Form](https://forms.gle/ALje6GSeh2okHbmx8)
- Multi-label Micro-Action-52 dataset (MMA-52): [Application Form](https://forms.gle/k9p7MxzEKT3iV27x6) and [LA File](https://drive.google.com/file/d/1uJ071OdsGKxWa70nOHdjDjnOfWXy7bgU/view?usp=sharing)


### MA-52 Statistics
<p align="center">
<img src="./assets/ma52.png" width="880">
</p>


### Micro-Action examples

For more micro-action samples, please refer to `MA-52 Dataset Samples.zip` in [huggingface](https://huggingface.co/datasets/kunli-cs/MA-52/tree/main). 

<table rules="none" align="center">
	<tr>
		<td>
			<center>
				<img src="./assets/ma52_demo/A1 shaking body/0030_01_0002.gif" width="100%" />
				<p>A1: shaking body</p>
      </center>
		</td>
		<td>
			<center>
				<img src="./assets/ma52_demo/A2 turning around/0078_01_0005.gif" width="100%" />
				<p>A2: turning around</p>
      </center>
		</td>
    <td>
			<center>
				<img src="./assets/ma52_demo/A3 sitting straightly/0020_01_0008.gif" width="100%" />
				<p>A3: sitting straightly</p>
      </center>
		</td>
  </tr>
  <tr>
		<td>
			<center>
				<img src="./assets/ma52_demo/B1 nodding/0019_02_0078.gif" width="100%" />
				<p>B1 nodding</p>
      </center>
		</td>
    <td>
			<center>
				<img src="./assets/ma52_demo/B2 shaking head/0035_02_0007.gif" width="100%" />
				<p>B2 shaking head</p>
			</center>
		</td>
    <td>
			<center>
				<img src="./assets/ma52_demo/B3 turning head/0010_02_0101.gif" width="100%" />
				<p>B3 turning head</p>
			</center>
		</td>
	</tr>
</table>

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
  pages={6238-6252},
  publisher={IEEE},
  doi={10.1109/TCSVT.2024.3358415}
}

@article{li2024mmad,
  title={MMAD: Multi-label Micro-Action Detection in Videos},
  author={Li, Kun and Guo, Dan and Liu, Pengyu and Chen, Guoliang and Wang, Meng},
  journal={arXiv preprint arXiv:2407.05311},
  year={2024}
}

@misc{MicroAction2024,
  author       = {Guo, Dan and Li, Kun and Hu, Bin and Zhang, Yan and Wang, Meng},
  title        = {Micro-Action Benchmark},
  year         = {2024},
  howpublished = {\url{https://github.com/VUT-HFUT/Micro-Action}},
  note         = {Accessed: 2024-08-21}
}

```
