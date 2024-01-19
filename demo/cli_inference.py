# created by Firok

import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

from demo import setup_cfg, get_parser

# constants
WINDOW_NAME = "COCO detections"

if __name__ == "__main__":
    from custom.cli_util import parser_addition, args_addition
    mp.set_start_method("spawn", force=True)
    parser = get_parser()
    parser_addition(parser)
    args = parser.parse_args()
    args_addition(args)

    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    while True:
        try:
            line = input('输入文件路径以推理: ').strip()
            if line == 'exit':
                break

            img = read_image(line, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    line,

                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                    )
            )

            print('读取 predictions')
            print(predictions)
            print('读取 instance')
            instances = predictions['instances']
            print('instances 数据')
            print(instances)

            print('scores')
            print(instances.scores)
            print('pred_masks')
            print(instances.pred_masks)
            print('pred_classes')
            print(instances.pred_classes)

            import imantics
            masks_raw = instances.pred_masks.cpu()
            for mask_raw in masks_raw:
                shape = imantics.Mask(mask_raw)
                print('shape')
                print(shape.polygons())


            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
            if cv2.waitKey(0) == 27:
                break  # esc to quit

        except Exception as e:
            print('发生错误')
            print(str(e))
