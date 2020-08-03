import torch
import numpy as np
import pickle
# numbers of class in dataset
num_class = 1
# load
with open('checkpoints/model_final_a54504.pkl', 'rb') as f:
    obj = f.read()
weights = pickle.loads(obj, encoding='latin1')
# modify mask r-cnn weight structure
weights['model']['roi_heads.box_predictor.cls_score.weight']=np.zeros([num_class+1,1024], dtype='float32')
weights['model']['roi_heads.box_predictor.cls_score.bias']=np.zeros([num_class+1], dtype='float32')

weights['model']['roi_heads.box_predictor.bbox_pred.weight']=np.zeros([num_class*4,1024], dtype='float32')
weights['model']['roi_heads.box_predictor.bbox_pred.bias']=np.zeros([num_class*4], dtype='float32')

weights['model']['roi_heads.mask_head.predictor.weight']=np.zeros([num_class,256,1,1], dtype='float32')
weights['model']['roi_heads.mask_head.predictor.bias']=np.zeros([num_class], dtype='float32')
# save
f = open('model_final_maskrcnn_fpn.pkl', 'wb')
pickle.dump(weights, f)
f.close()