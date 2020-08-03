# image augmentation
import numpy as np
import imgaug.augmenters as iaa
import cv2
import os, shutil

# sequential of augmentation method
seq = iaa.Sequential([
	iaa.SomeOf((1, 3),
		[
			iaa.OneOf([
				iaa.CoarseDropout(0.1, size_percent=0.1),
				iaa.CoarseDropout(0.1, size_percent=0.1, per_channel=1.0)
				]),
			iaa.AdditiveGaussianNoise(scale=0.2*255),
			iaa.SaltAndPepper(0.5),
			iaa.Multiply((0.2, 2.0)),
			iaa.Solarize(1.0, threshold=(32, 128)),
			iaa.GaussianBlur(sigma=(3.0, 6.0)),
			iaa.OneOf([
				iaa.imgcorruptlike.Snow(severity=4),
				iaa.imgcorruptlike.ZoomBlur(severity=2)
				]),
			iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.1, 0.8), per_channel=True)
		])
])

imgfold = os.listdir('D:/cards_dataset/traindata')
imglist = []
i=0
# load images
for ele in imgfold:
	img = cv2.imread('D:/cards_dataset/traindata/'+ele)
	imglist.append(img)
	if(i%100==0):
		print(i)
	i+=1
# do image augmentation
images_aug = seq(images=imglist)
cnt=0
# save augmented images and groundtruth
for ele in imgfold:
	cv2.imwrite("D:/cards_dataset/traindata/"+ele[:-4]+"_01.jpg",images_aug[cnt])
	shutil.copyfile("D:/cards_dataset/traindata_gt/"+ele[:-4]+'.json', "D:/cards_dataset/traindata_gt/"+ele[:-4]+'_01.json')
	cnt+=1