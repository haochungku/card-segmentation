import os
# check img and gt file are matched
imgdata = os.listdir('D:/cards_dataset/traindata/images')
gtdata = os.listdir('D:/cards_dataset/traindata/ground_truth')

for i, j in zip(imgdata, gtdata):
	if i[:-4]!=j[:-5]:
		print('err')
print('done')

# # rename to .tif if need
# testdata = os.listdir('D:/cards_dataset/testdata/images')
# traindata = os.listdir('D:/cards_dataset/traindata/images')

# for i in testdata:
# 	filenam = i[:-4]
# 	oldname = 'D:/cards_dataset/testdata/images/'+i
# 	newname = 'D:/cards_dataset/testdata/images/'+filenam+'.tif'
# 	os.rename(oldname, newname)

# for i in traindata:
# 	filenam = i[:-4]
# 	oldname = 'D:/cards_dataset/traindata/images/'+i
# 	newname = 'D:/cards_dataset/traindata/images/'+filenam+'.tif'
# 	os.rename(oldname, newname)
