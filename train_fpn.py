import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import datetime
import time
import os

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# training function
def Train():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # register train dataset
    register_coco_instances("custom", {}, "datasets/traindata/midv500_coco.json", "datasets/traindata/")
    custom_metadata = MetadataCatalog.get("custom")
    dataset_dicts = DatasetCatalog.get("custom")

    # set cfg
    cfg = get_cfg()
    cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
    cfg.DATASETS.TRAIN = ("custom",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = 'model_final_maskrcnn_fpn.pkl'
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.02
    cfg.SOLVER.MAX_ITER = (1000)  
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (512) 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 

    # training
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    Train()
