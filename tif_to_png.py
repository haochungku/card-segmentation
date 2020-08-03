# convert .fie to .png if need
# save all data into only 2 folder (images and groundtruth)
import os, sys, shutil
from PIL import Image

datafold = os.listdir('D:/midv500')
senriao = ['CA', 'CS', 'HA', 'HS', 'KA', 'KS', 'PA', 'PS', 'TA', 'TS']

for ele in datafold:
	for ele2 in senriao:
		imgfolder = 'D:/midv500/'+ele+'/images/'+ele2
		gtfolder = 'D:/midv500/'+ele+'/ground_truth/'+ele2
		imgpath=os.listdir(imgfolder)
		for ele3 in imgpath:
			im = Image.open(imgfolder+'/'+ele3)
			# convert img
			im.save('D:/cards_dataset/traindata/'+ele+'_'+ele2+'_'+ele3[:-4]+'.png')
			im.close()
			# copy gt
			shutil.copyfile(gtfolder+'/'+ele3[:-4]+'.json', 'D:/cards_dataset/traindata_gt/'+ele+'_'+ele2+'_'+ele3[:-4]+'.json')