# https://arxiv.org/abs/1611.07004  ##pix2pix
# https://arxiv.org/abs/1505.04597  ##Unet

from pix2pix import pix2pix
import image_functions as img

import tensorflow as tf #version 1.4
import scipy
import numpy as np
import os

trainset_path = './edges2shoes/train/'
valiset_path = './edges2shoes/val/'

saver_path = './saver/'
make_image_path = './generate/'

batch_size = 4

def train(model, train_set, epoch):
	total_D_loss = 0
	total_G_loss = 0

	iteration = int(np.ceil(len(train_set)/batch_size))

	for i in range( iteration ):
		print('epoch', epoch, 'batch', i+1,'/', iteration )

		filelist = train_set[batch_size * i : batch_size * (i + 1)]

		#batch image read, preprocess, split_A_B
		batch = []
		for name in filelist:
			batch.append(img.image_read(trainset_path+name))
		batch = img.image_preprocess(np.array(batch, dtype=np.float32))
		X, Y = img.image_split_A_B(batch)
		###############

		#Discriminator 학습.
		_, D_loss = sess.run([model.D_minimize, model.D_loss], {
						model.X:X, model.Y:Y, model.is_train:True
					}
				)
		
		#Generator 학습. 		
		_, G_loss = sess.run([model.G_minimize, model.G_loss], {
						model.X:X, model.Y:Y, model.is_train:True
					}
				)
		
		#parameter sum
		total_D_loss += D_loss
		total_G_loss += G_loss

	return total_D_loss/iteration, total_G_loss/iteration



def write_tensorboard(model, D_loss, G_loss, epoch):
	summary = sess.run(model.merged, 
					{
						model.D_loss_tensorboard:D_loss, 
						model.G_loss_tensorboard:G_loss,
					}
				)

	model.writer.add_summary(summary, epoch)



def gen_image(model, vali_set, epoch):
	path = make_image_path+str(epoch)+'/'
	if not os.path.exists(make_image_path+str(epoch)+'/'):
		os.makedirs(make_image_path+str(epoch)+'/')
		
	for name in vali_set:
		vali_image = img.image_read(valiset_path+name)
		vali_image = img.image_preprocess(np.array(vali_image, dtype=np.float32))
		X, Y = img.image_split_A_B(vali_image, 1) # 256 256 3, 256 256 3


		generated = sess.run(model.Gen, { # 1 256 256 3
						model.X:[X], model.is_train:False	
					}
				)

		concat = np.concatenate((X, Y, generated[0]), axis=1)
		img.image_save(path+name, concat)



def run(model, train_set, vali_set, restore = 0):
	#restore인지 체크.
	if restore != 0:
		model.saver.restore(sess, saver_path+str(restore)+".ckpt")
	
	print('training start')


	#학습 진행
	for epoch in range(restore + 1, 2001):
		D_loss, G_loss = train(model, train_set, epoch)

		print("epoch : ", epoch, " D_loss : ", D_loss, " G_loss : ", G_loss)

		
		if epoch % 3 == 0:
			#tensorboard
			write_tensorboard(model, D_loss, G_loss, epoch)

			#weight 저장할 폴더 생성
			if not os.path.exists(saver_path):
				os.makedirs(saver_path)
			save_path = model.saver.save(sess, saver_path+str(epoch)+".ckpt")
		
			#생성된 이미지 저장할 폴더 생성
			if not os.path.exists(make_image_path):
				os.makedirs(make_image_path)
			gen_image(model, vali_set, epoch)




sess = tf.Session()

#model
model = pix2pix(sess) 


#필요한 batch만큼 디스크에서 읽어 오려고 파일 이름만 가져옴.
train_set = img.get_image_filelist(trainset_path)
vali_set = img.get_image_filelist(valiset_path)
#print(train_set.shape, vali_set.shape)

run(model, train_set, vali_set)

