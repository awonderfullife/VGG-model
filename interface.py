import os
import argparse
import sys
import VGG_model
import cv2
import tensorflow as tf 
import numpy as np 
import item_classes

parser = argparse.ArgumentParser(description="Classify some images")
parser.add_argument('path', default="testImages")
args = parser.parse_args(sys.argv[1:])


withPath = lambda f: '{}/{}'.format(args.path, f)
testImg = dict((f, cv2.imread(withPath(f))) for f in os.listdir(args.path) if os.path.isfile(withPath(f)))

if testImg.values():
	dropoutPro = 1
	classNum = 1000
	skip = []

	imgMean = np.array([104, 117, 124], np.float)
	x = tf.placeholder("float", [1, 224, 224, 3])

	model = VGG_model.VGG_19(x, dropoutPro, classNum, skip)
	score = model.fc8
	softmax = tf.nn.softmax(score)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		model.loadModel(sess)

		for key, img in testImg.items():
			resized_img = cv2.resize(img.astype(np.float), (224, 224)) - imgMean
			maxx = np.argmax(sess.run(softmax, feed_dict={x: resized_img.reshape((1, 224, 224, 3))}))
			res = item_classes.class_names[maxx]

			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(img, res, (int(img.shape[0]/3), int(img.shape[1]/3)), font, 1, (0, 255, 0), 2)
			print "{}: {}\n --- ".format(key, res)
			cv2.imwrite("{}_{}.jpg".format(key,res), img)



