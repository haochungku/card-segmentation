# convert dataset to coco format
import midv500
dataset_dir = 'D:/cards_dataset/traindata/'
export_dir = 'D:/cards_dataset/traindata/'
midv500.convert_to_coco(dataset_dir, export_dir)

dataset_dir = 'D:/cards_dataset/testdata/'
export_dir = 'D:/cards_dataset/testdata/'
midv500.convert_to_coco(dataset_dir, export_dir)

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
