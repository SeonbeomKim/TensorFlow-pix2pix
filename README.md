# TensorFlow-pix2pix
Image-to-Image Translation with Conditional Adversarial Network

## paper  
https://arxiv.org/abs/1611.07004 pix2pix
![testImage](./pix2pix_paper_image/pix2pix_object.png)

https://arxiv.org/abs/1505.04597 Unet

## Receptive field simulation
https://fomoro.com/tools/receptive-fields/#4,2,1,SAME;4,2,1,SAME;4,2,1,SAME;4,1,1,SAME;4,1,1,SAME 70x70

## Env
TensorFlow version == 1.4
GTX-1080TI

## dataset
edges2shoes

## pix2pix.py
pix2pix가 구현되어 있는 Class

## pix2pix_train.py
pix2pix를 학습시키는 코드

## image_functions.py
pix2pix 코드에 사용되는 이미지 관련 함수들 정의

## saver
epoch 45 동안 학습된 파라미터
zip 형식으로 분할 압축 되어 있음(7-Zip 이용)

## generate
학습된 결과 shoes 생성 이미지(epoch 45 파라미터 이용)


## pix2pix edges2shoes result (after 45 epochs of training) 
Input / Ground Truth / Generated

![testImage](./generate/45/48.jpg)
![testImage](./generate/45/51.jpg)
![testImage](./generate/45/73.jpg)
![testImage](./generate/45/79.jpg)
![testImage](./generate/45/93.jpg)
![testImage](./generate/45/187.jpg)
![testImage](./generate/45/190.jpg)
![testImage](./generate/45/200.jpg)
![testImage](./generate/45/181.jpg)
![testImage](./generate/45/177.jpg)
![testImage](./generate/45/176.jpg)
![testImage](./generate/45/125.jpg)
![testImage](./generate/45/117.jpg)
![testImage](./generate/45/104.jpg)
