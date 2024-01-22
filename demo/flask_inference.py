# 用 Flask 开一个服务端接口
# 使用这个脚本需要先 pip install Flask

import multiprocessing as mp
import time
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

from demo import setup_cfg, get_parser

from flask import Flask, request, jsonify

# copy from AlloyWrench 0.36.0
# https://github.com/FirokOtaku/AlloyWrench/blob/main/src/main/python/firok/tool/alloywrench/convert_result.py

from matplotlib.patches import Polygon
import torch
import numpy

def convert(target):
    type_target = type(target)
    if type_target == tuple or type_target == list:
        ret = []
        for tuple_child in target:
            temp_child = convert(tuple_child)
            ret.append(temp_child)
        return ret
    elif type_target == numpy.ndarray:
        ret = []
        for tuple_child in target.tolist():
            temp_child = convert(tuple_child)
            ret.append(temp_child)
        return ret
    elif type_target == dict:
        ret = {}
        for (key, value) in target.items():
            ret[key] = convert(value)
        return ret
    elif type_target == Polygon:
        return convert(target.xy)
    elif type_target == torch.Tensor:
        if target.dim() == 0:
            return target.item()
        ret = []
        for list_child in target:
            ret.append(convert(list_child))
        return ret
    else:
        return target


import imantics, json

app = Flask('detectron2_service')

@app.route('/inference', methods=['POST'])
def inference():
    result = {
        'startTime': time.time(),
    }

    local_image_path = request.form['path']

    img = read_image(local_image_path, format="BGR")
    predictions, visualized_output = demo.run_on_image(img)

    instances = predictions['instances']
    result['scores'] = instances.scores
    result['pred_classes'] = instances.pred_classes
    result['pred_shapes'] = []
    for mask_raw in instances.pred_masks.cpu():
        shape = imantics.Mask(mask_raw)
        shape_polygon = shape.polygons()
        pred_shape = []
        for shape_polygon_pt in shape_polygon:
            pred_shape.append(shape_polygon_pt)
        result['pred_shapes'].append(pred_shape)

    result['endTime'] = time.time()
    result['costTime'] = result['endTime'] - result['startTime']

    return json.dumps(convert(result))


if __name__ == "__main__":

    print('Detectron2 推理服务启动...')
    from custom.cli_util import parser_addition, args_addition
    mp.set_start_method("spawn", force=True)
    parser = get_parser()
    parser_addition(parser)
    parser.add_argument("--flask-port", type=int, required=True, help="port to run Flask service")

    args = parser.parse_args()
    args_addition(args)
    flask_port = args.flask_port

    setup_logger(name="fvcore")

    print('加载推理配置...')
    cfg = setup_cfg(args)
    print('加载模型...')
    demo = VisualizationDemo(cfg)

    import logging
    # 禁用 Flask 日志输出
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    print('启动服务端接口...')
    app.run(host='0.0.0.0', port=flask_port)
