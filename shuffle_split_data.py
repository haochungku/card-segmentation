# shuffle dataset and split into 'traindata' and 'testdata' dataset (70% 30%)
import os, random, shutil
imgdata = os.listdir('D:/cards_dataset/traindata')
random.shuffle(imgdata)
cnt = int(len(imgdata)*0.3)

for i in range(0, cnt):
	imgfile = imgdata[i]
	gtfile = imgfile[:-4]+'.json'
	shutil.move('D:/cards_dataset/traindata/'+imgfile,'D:/cards_dataset/testdata')
	shutil.move('D:/cards_dataset/traindata_gt/'+gtfile,'D:/cards_dataset/testdata_gt')