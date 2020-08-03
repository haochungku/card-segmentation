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

# prediction and evaluation function
def Predict():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # register test dataset
    register_coco_instances("custom", {}, "datasets/testdata/midv500_coco.json", "datasets/testdata/")
    custom_metadata = MetadataCatalog.get("custom")
    dataset_dicts = DatasetCatalog.get("custom")

    # set cfg
    cfg = get_cfg()
    cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
    cfg.DATASETS.TEST = ("custom", )
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (512)  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 
    predictor = DefaultPredictor(cfg)

    # save prediction image results
    cnt=0
    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], metadata=custom_metadata, scale=1, instance_mode=ColorMode.IMAGE_BW)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite('D:/eagletmp/detectron2-maskrcnn/outputimg/'+str(cnt)+'.png',v.get_image()[:, :, ::-1])
        cnt+=1

    # model evaulation
    evaluator = COCOEvaluator("custom", cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "custom")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))


if __name__ == "__main__":
    Predict()
