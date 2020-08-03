"# card-segmentation" 
### The numbers are execution order.

## Image Preprocessing 

1.tif_to_png.py
* This is used to let midv500 dataset more simple.
  This file let the dataset only have two folders, "traindata" to save all images, "traindata_gt" to save all corresponding groundtruth.
  So, image and groundtruth will be renamed.

2.augmentation.py
* This is used to augment all images in folder "traindata".
  folder "traindata" now contains all images in dataset.
  folder "traindata" now remains zero images.

3.shuffle_split_data.py
* This is used to shuffle all images and split into training dataset and test dataset.
  First, shuffle all images in folder "traindata".
  Then, split all images by 70%, 30% into folder "traindata" and folder "testdata".

4.convert_to_coco_format.py
* This is used to convert our dataset into "coco-format" dataset.
  Since our dataset is midv500, we use midv500's function to convert.
  However, it seems to need .tif format, you can use like rename method in comment.

## Model Training

1.changemaskrcnn_fpn.py
* This is used to modify mask r-cnn weight structure.
  Since our dataset only have one class, pre-train models we use need to change some weights.
  Change variable "num_class" into number of class in dataset, that is 1.
  changemaskrcnn_c4.py and changemaskrcnn_dc5.py are same, just different .pkl file.

2.train_fpn.py
* This is used to train model.
  First, register our training dataset.
  Second, set cfg.
  Last, start to train.
  train_c4.py and train_dc5.py are same, just different .yaml and .pkl file.

## Model Predicting

1.predict_fpn.py
* This is used to predict model.
  First, register our testing dataset.
  Second, set cfg.
  Then, predict images and save the results into new images.
  Last, evaluate model performance by testing dataset.
  predict_c4.py and predict_dc5.py are same, just different .yaml and .pth file.
  
## Model Download

* Models including pre-train models in Detectron2 Model Zoo ("checkpoints" folder), change mask r-cnn structure models (model_final_maskrcnn_XXX.pkl) and training results models ("output_XXX_batch_2_sizeimage_512" folder).
* Models are in https://drive.google.com/file/d/1ekt3Klj7BPI3t8NHgxSHdnqIntgtGWwr/view?usp=sharing
* Image augmentation samples are also included.
