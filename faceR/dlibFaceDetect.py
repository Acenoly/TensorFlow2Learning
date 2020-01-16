#coding:utf-8
import cv2
import dlib
import sys
import numpy as np
import os

#获取当前路径
def cut_face(path, save_path):
	predictor_path = 'shape_predictor_68_face_landmarks.dat'
	face_file_path = path
	print(face_file_path)
	#导入人脸检测模型
	detector = dlib.get_frontal_face_detector()
	#导入检测人脸特征点的模型
	sp = dlib.shape_predictor(predictor_path)
	#读入图片
	bgr_img = cv2.imread(face_file_path)
	if bgr_img is None:
		print("img no exist")
		exit()

	#opencv的颜色空间是BGR，需要转为RGB才能用在dlib中
	rgb_img = cv2.cvtColor(bgr_img,cv2.COLOR_BGR2RGB)
	#检测图片中的人脸
	dets = detector(rgb_img,1)
	num_faces = len(dets)
	if num_faces == 0:
		print("no face")
		return

	#识别人脸特征点，并保存下来
	faces = dlib.full_object_detections()
	for det in dets:
		faces.append(sp(rgb_img,det))

	#人脸对齐
	images = dlib.get_face_chips(rgb_img,faces,size=320)
	#显示计数，按照这个计数创建窗口
	image_cnt = 0
	#显示对齐结果
	for image in images:
		image_cnt += 1
		cv_rgb_image = np.array(image).astype(np.uint8)  #先转化为numpy数组
		cv_bgr_image = cv2.cvtColor(cv_rgb_image,cv2.COLOR_RGB2BGR)
		cv2.imwrite(save_path, cv_bgr_image)


def mkdir(path):
	folder = os.path.exists(path)

	if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
	else:
		pass


#文件夹下子文件
filenames=os.listdir(r'G:\facedata')
print(filenames)

for dirs in filenames:
	pics = os.listdir(r'G:\facedata' + "\\" + dirs)
	mkdir(r'G:\facedatahandle' + "\\" + dirs)
	for pic in pics:
		pic_path_cut = r'G:\facedatahandle' + "\\" + dirs + "\\" + "cut_" + pic.replace("bmp", "jpg")
		pic_path = r'G:\facedata' + "\\" + dirs + "\\" + pic
		cut_face(pic_path, pic_path_cut)
		print(pic_path_cut)
