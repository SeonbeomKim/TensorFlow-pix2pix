import numpy as np
import tensorflow as tf
import scipy
import os

def image_save(path, image):
	scipy.misc.imsave(path, image)


def image_read(path):
	image = scipy.misc.imread(path)
	return image


def image_preprocess(image):
	# 0<= images <= 255
	# pix2pix 에서는 tanh로 생성해서 비교하기 때문에 -1 <= images <= 1로 맞춰줘야함.
	image /= 255 # 0 ~ 1
	image -= 0.5 # -0.5 ~ 0.5
	image /= 0.5 # -1 ~ 1
	return image

'''
#메모리에 모든 이미지 올려놓는 코드
def get_image_from_folder(path):
	images = []
	for path, _, filelist in os.walk(path):
		for files in filelist:
			#print(path+files)
			images.append(image_read(path+files))

	return image_preprocess(np.array(images, dtype=np.float32))
'''

#메모리 부족해서 필요한 이미지만 그때그때 메모리에 올리기 위해 path만 저장하는 코드.
def get_image_filelist(path):
	filelist = list(os.walk(path))[0][2]
	return np.array(filelist)


def image_split_A_B(image, batchsize=0):
	if batchsize == 1:
		image_A = image[:, :256, :]
		image_B = image[:, 256:, :]
		return image_A, image_B
	else:
		image_A = image[:, :, :256, :]
		image_B = image[:, :, 256:, :]
		return image_A, image_B

