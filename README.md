# HSI-Object-Detection-NPU
We have released our dataset proposed in paper 'Object Detection in Hyperspectral Images'. 
Raw hyperspectral images and processed data (96-channel) can be found at [[baidu cloud]( https://pan.baidu.com/s/1mtXDJfU6M8F60GZinLam-w), password: 6shr],
[[Onedrive](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/yanlongbin_mail_nwpu_edu_cn/ERsB07TPh8RGrNpsgIejn38B0rmwzJEBgLmL5hzwvYlV7g?e=Upk6iW)].

If you use these datasets in your researches or works, please cite our paper as follow:

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
1.Training

generate label json file
```
python create_data_lists.py
```

then

```
python train.py
```

2.eval

```
python eval.py
```

pretrained model download: [[baidu cloud](https://pan.baidu.com/s/11mQsR10Z35EH6Kw9__LyrA), password: 5s97]
