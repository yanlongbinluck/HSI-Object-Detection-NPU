# HSI-Object-Detection-NPU
We have released our dataset proposed in paper 'Object Detection in Hyperspectral Images'. 
Raw hyperspectral images and processed data (96-channel) can be found at [[baidu cloud]( https://pan.baidu.com/s/1mtXDJfU6M8F60GZinLam-w), password: 6shr],
[[Onedrive](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/yanlongbin_mail_nwpu_edu_cn/ERsB07TPh8RGrNpsgIejn38B0rmwzJEBgLmL5hzwvYlV7g?e=Upk6iW)].

![](https://github.com/yanlongbinluck/HSI-Object-Detection-NPU/blob/main/fig/results.png)

1.Main libraries

```
torch==1.1.0
cuda==10.0
libtiff==0.4.2
```

2.Training

generate label json file
```
python create_data_lists.py
```

then

```
python train.py
```
Note that due to samll scale of training dataset, the mAP may have relatively large jitters (about 2 mAP) with different random seeds.

3.Eval

```
python eval.py
```

our pretrained model: [baidu cloud](https://pan.baidu.com/s/11mQsR10Z35EH6Kw9__LyrA)

4.Acknowledgements

Our work is implemented based on [this](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection) repo, thanks for this work.

If you use our work in your researches, please cite our paper as follow:

```
@article{yan2021object,
  title={Object Detection in Hyperspectral Images},
  author={Yan, Longbin and Zhao, Min and Wang, Xiuheng and Zhang, Yuge and Chen, Jie},
  journal={IEEE Signal Processing Letters},
  volume={28},
  pages={508--512},
  year={2021},
  publisher={IEEE}
}
```
