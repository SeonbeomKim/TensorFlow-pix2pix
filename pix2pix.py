# https://arxiv.org/abs/1611.07004  ##pix2pix
# https://fomoro.com/tools/receptive-fields/#4,2,1,SAME;4,2,1,SAME;4,2,1,SAME;4,1,1,SAME;4,1,1,SAME  ##patch(receptive field) size simulation
# https://arxiv.org/abs/1505.04597  ##Unet

import tensorflow as tf #version 1.4
import numpy as np

class pix2pix:

	def __init__(self, sess):
		self.train_rate = 0.0002
		self.channel = 3 
		self.height = 256 
		self.width = 256
		self.L1_lambda = 100
			

		with tf.name_scope("placeholder"):
			#input
			self.X = tf.placeholder(tf.float32, [None, self.height, self.width, self.channel])
			#target
			self.Y = tf.placeholder(tf.float32, [None, self.height, self.width, self.channel])
			#batch_norm
			self.is_train = tf.placeholder(tf.bool)



		#input X로부터 데이터 생성. drop out이 noise의 역할을 대신함. 기존처럼 noise를 붙이는경우 noise를 무시하는 방향으로 학습한다고 논문에 나옴.
		with tf.name_scope("generate_image"):
			self.Gen = self.Generator(self.X) #batch_size, self.height, self.width, self.channel
			#self.Gen은 tanh 한 결과임. -1~1


		#Discriminator가 진짜라고 생각하는 확률		
		with tf.name_scope("result_from_Discriminator"):
			#원본 입력 및 정답으로부터 항상 참이라고 배울 부분
			self.D_X_logits = self.Discriminator(self.X, self.Y) 
			
			#D 학습시: 원본 입력 및 생성된 정답으로부터 항상 거짓이라고 배울 부분.
			#G 학습시: 원본 입력 및 생성된 정답으로부터 항상 참이라고 배울 부분.
			self.D_Gen_logits = self.Discriminator(self.X, self.Gen, True)



		with tf.name_scope("loss"):
			#Discriminator 입장에서 최소화 해야 하는 값
			self.D_loss = self.Discriminator_loss_function(self.D_X_logits, self.D_Gen_logits)
			#Generator 입장에서 최소화 해야 하는 값.
			self.G_loss = self.Generator_loss_function(self.D_Gen_logits, self.Gen, self.Y)



		#학습 코드
		with tf.name_scope("train"):
			#Batch norm 학습 방법 : https://www.tensorflow.org/versions/r1.4/api_docs/python/tf/layers/batch_normalization
			with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
					
					#Discriminator와 Generator, Q에서 사용된 variable 분리.
				self.D_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Discriminator')
				self.G_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Generator')
				
					
				self.D_minimize = tf.train.AdamOptimizer(
							learning_rate=self.train_rate, beta1=0.5).minimize(self.D_loss, var_list=self.D_variables) #D 변수만 학습.
				self.G_minimize = tf.train.AdamOptimizer(
							learning_rate=self.train_rate, beta1=0.5).minimize(self.G_loss, var_list=self.G_variables) #G 변수만 학습.
				



		#tensorboard
		with tf.name_scope("tensorboard"):
			self.D_loss_tensorboard = tf.placeholder(tf.float32) #Discriminator 입장에서 최소화 해야 하는 값
			self.G_loss_tensorboard = tf.placeholder(tf.float32) #Generator 입장에서 최소화 해야 하는 값.

			self.D_loss_summary = tf.summary.scalar("D_loss", self.D_loss_tensorboard) 
			self.G_loss_summary = tf.summary.scalar("G_loss", self.G_loss_tensorboard) 
			
			self.merged = tf.summary.merge_all()
			self.writer = tf.summary.FileWriter('./tensorboard/', sess.graph)



		with tf.name_scope("saver"):
			self.saver = tf.train.Saver(max_to_keep=10000)



		sess.run(tf.global_variables_initializer())




	#translation 하고 싶은 이미지(data)와, 정답(target)을 입력으로 함.
	def Discriminator(self, data, target, reuse=False): #batch_size, 32, 32, 1

		with tf.variable_scope('Discriminator') as scope:
			if reuse == True: #Descriminator 함수 두번 부르는데 두번째 부르는 때에 같은 weight를 사용하려고 함.
				scope.reuse_variables()

			# channel 단위에서의 concat. ex)(batch,256,256,3) || (batch,256,256,3) == (batch,256,256,6)
			XY = tf.concat((data, target), axis=-1) 

			#input layer는 BN 안함.
			#receptive field 4
			D_conv1 = tf.layers.conv2d(inputs=XY, filters=64, kernel_size=[4, 4], strides=(2, 2), padding='same') #batch, 128, 128, 64
			D_conv1 = tf.nn.leaky_relu(D_conv1) # default leak is 0.2

			#receptive field 10
			D_conv2 = tf.layers.conv2d(inputs=D_conv1, filters=128, kernel_size=[4, 4], strides=(2, 2), padding='same') #batch, 64, 64, 128
			D_conv2 = tf.layers.batch_normalization(D_conv2, training=self.is_train)
			D_conv2 = tf.nn.leaky_relu(D_conv2)

			#receptive field 22
			D_conv3 = tf.layers.conv2d(inputs=D_conv2, filters=256, kernel_size=[4, 4], strides=(2, 2), padding='same') #batch, 32, 32, 256
			D_conv3 = tf.layers.batch_normalization(D_conv3, training=self.is_train)
			D_conv3 = tf.nn.leaky_relu(D_conv3)

			#receptive field 46
			D_conv4 = tf.layers.conv2d(inputs=D_conv3, filters=512, kernel_size=[4, 4], strides=(1, 1), padding='same') #batch, 32, 32, 512
			D_conv4 = tf.layers.batch_normalization(D_conv4, training=self.is_train)
			D_conv4 = tf.nn.leaky_relu(D_conv4)

			#receptive field 70
			D_conv5_logits = tf.layers.conv2d(inputs=D_conv4, filters=1, kernel_size=[4, 4], strides=(1, 1), padding='same') #batch, 32, 32, 1
			#D_conv5 = tf.layers.batch_normalization(D_conv5, training=self.is_train)
			#D_conv5 = tf.nn.sigmoid(D_conv5)


		return D_conv5_logits



	#translation 하고 싶은 이미지(data) 입력. noise의 역할은 drop-out이 대신함.
	def Generator(self, data): #batch_size, self.height, self.width, self.channel
		

		with tf.variable_scope('Generator'):
			with tf.name_scope("Encoder"):
				#input layer는 BN 안함.
				G_conv1 = tf.layers.conv2d(inputs=data, filters=64, kernel_size=[4, 4], strides=(2, 2), padding='same') #batch, 128, 128, 64
				G_conv1_lrelu = tf.nn.leaky_relu(G_conv1) # default leak is 0.2

				G_conv2 = tf.layers.conv2d(inputs=G_conv1_lrelu, filters=128, kernel_size=[4, 4], strides=(2, 2), padding='same') #batch, 64, 64, 128
				G_conv2 = tf.layers.batch_normalization(G_conv2, training=self.is_train)
				G_conv2_lrelu = tf.nn.leaky_relu(G_conv2)

				G_conv3 = tf.layers.conv2d(inputs=G_conv2_lrelu, filters=256, kernel_size=[4, 4], strides=(2, 2), padding='same') #batch, 32, 32, 256
				G_conv3 = tf.layers.batch_normalization(G_conv3, training=self.is_train)
				G_conv3_lrelu = tf.nn.leaky_relu(G_conv3)

				G_conv4 = tf.layers.conv2d(inputs=G_conv3_lrelu, filters=512, kernel_size=[4, 4], strides=(2, 2), padding='same') #batch, 16, 16, 512
				G_conv4 = tf.layers.batch_normalization(G_conv4, training=self.is_train)
				G_conv4_lrelu = tf.nn.leaky_relu(G_conv4)

				G_conv5 = tf.layers.conv2d(inputs=G_conv4_lrelu, filters=512, kernel_size=[4, 4], strides=(2, 2), padding='same') #batch, 8, 8, 512
				G_conv5 = tf.layers.batch_normalization(G_conv5, training=self.is_train)
				G_conv5_lrelu = tf.nn.leaky_relu(G_conv5)

				G_conv6 = tf.layers.conv2d(inputs=G_conv5_lrelu, filters=512, kernel_size=[4, 4], strides=(2, 2), padding='same') #batch, 4, 4, 512
				G_conv6 = tf.layers.batch_normalization(G_conv6, training=self.is_train)
				G_conv6_lrelu = tf.nn.leaky_relu(G_conv6)

				G_conv7 = tf.layers.conv2d(inputs=G_conv6_lrelu, filters=512, kernel_size=[4, 4], strides=(2, 2), padding='same') #batch, 2, 2, 512
				G_conv7 = tf.layers.batch_normalization(G_conv7, training=self.is_train)
				G_conv7_lrelu = tf.nn.leaky_relu(G_conv7)

				G_conv8 = tf.layers.conv2d(inputs=G_conv7_lrelu, filters=512, kernel_size=[4, 4], strides=(2, 2), padding='same') #batch, 1, 1, 512
				G_conv8 = tf.layers.batch_normalization(G_conv8, training=self.is_train)
				G_conv8_lrelu = tf.nn.leaky_relu(G_conv8)


			with tf.name_scope("U-Net_decoder"):
				G_upconv1 = tf.layers.conv2d_transpose(inputs=G_conv8_lrelu, filters=512, kernel_size=[4, 4], strides=(2, 2), padding='same') #batch, 2, 2, 512
				G_upconv1 = tf.layers.batch_normalization(G_upconv1, training=self.is_train)
				G_upconv1 = tf.nn.dropout(G_upconv1, 0.5)
				G_upconv1 = tf.concat((G_upconv1, G_conv7), axis=-1) # batch, 2, 2, 1024
				G_upconv1_relu = tf.nn.relu(G_upconv1)

				G_upconv2 = tf.layers.conv2d_transpose(inputs=G_upconv1_relu, filters=1024, kernel_size=[4, 4], strides=(2, 2), padding='same') #batch, 4, 4, 1024
				G_upconv2 = tf.layers.batch_normalization(G_upconv2, training=self.is_train)
				G_upconv2 = tf.nn.dropout(G_upconv2, 0.5)
				G_upconv2 = tf.concat((G_upconv2, G_conv6), axis=-1) # batch, 4, 4, 1024+512
				G_upconv2_relu = tf.nn.relu(G_upconv2)

				G_upconv3 = tf.layers.conv2d_transpose(inputs=G_upconv2_relu, filters=1024, kernel_size=[4, 4], strides=(2, 2), padding='same') #batch, 8, 8, 1024
				G_upconv3 = tf.layers.batch_normalization(G_upconv3, training=self.is_train)
				G_upconv3 = tf.nn.dropout(G_upconv3, 0.5)
				G_upconv3 = tf.concat((G_upconv3, G_conv5), axis=-1) # batch, 8, 8, 1024+512
				G_upconv3_relu = tf.nn.relu(G_upconv3)
				
				#여기부터 dropout X
				G_upconv4 = tf.layers.conv2d_transpose(inputs=G_upconv3_relu, filters=1024, kernel_size=[4, 4], strides=(2, 2), padding='same') #batch, 16, 16, 1024
				G_upconv4 = tf.layers.batch_normalization(G_upconv4, training=self.is_train)
				G_upconv4 = tf.concat((G_upconv4, G_conv4), axis=-1) # batch, 16, 16, 1024+512
				G_upconv4_relu = tf.nn.relu(G_upconv4)

				G_upconv5 = tf.layers.conv2d_transpose(inputs=G_upconv4_relu, filters=1024, kernel_size=[4, 4], strides=(2, 2), padding='same') #batch, 32, 32, 1024
				G_upconv5 = tf.layers.batch_normalization(G_upconv5, training=self.is_train)
				G_upconv5 = tf.concat((G_upconv5, G_conv3), axis=-1) # batch, 32, 32, 1024+256
				G_upconv5_relu = tf.nn.relu(G_upconv5)

				G_upconv6 = tf.layers.conv2d_transpose(inputs=G_upconv5_relu, filters=512, kernel_size=[4, 4], strides=(2, 2), padding='same') #batch, 64, 64, 512
				G_upconv6 = tf.layers.batch_normalization(G_upconv6, training=self.is_train)
				G_upconv6 = tf.concat((G_upconv6, G_conv2), axis=-1) # batch, 64, 64, 512+128
				G_upconv6_relu = tf.nn.relu(G_upconv6)

				G_upconv7 = tf.layers.conv2d_transpose(inputs=G_upconv6_relu, filters=256, kernel_size=[4, 4], strides=(2, 2), padding='same') #batch, 128, 128, 256
				G_upconv7 = tf.layers.batch_normalization(G_upconv7, training=self.is_train)
				G_upconv7 = tf.concat((G_upconv7, G_conv1), axis=-1) # batch, 128, 128, 256+64
				G_upconv7_relu = tf.nn.relu(G_upconv7)

				G_upconv8 = tf.layers.conv2d_transpose(inputs=G_upconv7_relu, filters=128, kernel_size=[4, 4], strides=(2, 2), padding='same') #batch, 256, 256, 128
				G_upconv8 = tf.layers.batch_normalization(G_upconv8, training=self.is_train)
				G_upconv8_relu = tf.nn.relu(G_upconv8)

				G_output = tf.layers.conv2d_transpose(inputs=G_upconv8_relu, filters=3, kernel_size=[1, 1], strides=(1, 1), padding='same') #batch, 256, 256, 3
				#G_upconv8 = tf.layers.batch_normalization(G_upconv8, training=self.is_train)
				G_output_tanh = tf.nn.tanh(G_output)
				

			return G_output_tanh


	
	#Discriminator 학습.
	def Discriminator_loss_function(self, D_X_logits, D_Gen_logits):
		#return tf.reduce_mean(tf.log(D_X) + tf.log(1-D_Gen)) 기존 코드.		
		#위 식이 최대화가 되려면 D_X가 1이 되어야 하며, D_Gen이 0이 되어야 한다.
		#tf.ones_like(X) X와 같은 shape의 1로 이루어진 tensor를 리턴. D_X_logits을 sigmoid 한 결과와 1의 오차.
		D_X_loss = tf.nn.sigmoid_cross_entropy_with_logits(
					labels=tf.ones_like(D_X_logits), 
					logits=D_X_logits
				)

		D_Gen_loss = tf.nn.sigmoid_cross_entropy_with_logits(
					labels=tf.zeros_like(D_Gen_logits),
					logits=D_Gen_logits
				)

		#이 두 오차의 합을 최소화 하도록 학습.
		D_loss = (tf.reduce_mean(D_X_loss) + tf.reduce_mean(D_Gen_loss)) / 2

		return D_loss



	#Generator 입장에서 최소화 해야 하는 값.
	def Generator_loss_function(self, D_Gen_logits, Gen, Y): #Gen: generator가 생성한것. Y: 실제 target
		#return tf.reduce_mean(tf.log(D_Gen))
		#위 식이 최대화가 되려면 D_Gen이 1이 되어야 함. == 1과의 차이를 최소화 하도록 학습하면 됨.
		G_loss = tf.nn.sigmoid_cross_entropy_with_logits(
					labels=tf.ones_like(D_Gen_logits), 
					logits=D_Gen_logits
				)
		
		G_loss = tf.reduce_mean(G_loss) + self.L1_lambda * tf.reduce_mean(tf.abs(Gen - Y)) #L1 loss

		return G_loss
