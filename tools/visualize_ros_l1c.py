import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as Image2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import ros_numpy
import gc
import os
import sys
import argparse
import copy
import os
import time
import copy

import mmcv
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.parallel import DataContainer as DC
import torch
from mmcv.runner import load_checkpoint
from torchpack import distributed as dist
from torchpack.utils.config import configs
import tqdm

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmcv.runner import wrap_fp16_model
import os
cwd = os.getcwd()
from mmdet3d.core.bbox.structures.box_3d_mode import (
        Box3DMode,
        CameraInstance3DBoxes,
        DepthInstance3DBoxes,
        LiDARInstance3DBoxes,
    )

def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj

#dist.init()
config = "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml"
mode = "pred"
checkpoint = "pretrained/bevfusion-det.pth"
split = "val"
bbox_classes = None
bbox_score = 0.1
map_score = 0.5
out_dir = "results/visualize"
img_pub = []

configs.load(config, recursive=True)

cfg = Config(recursive_eval(configs), filename=config)
cfg.dist_params = dict(backend='nccl')
distributed = False
#torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
#torch.cuda.set_device(dist.local_rank())

# build the dataloader
dataset = build_dataset(cfg.data[split])
dataflow = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=True,
    shuffle=False,
)

# build the model and load checkpoint
model = build_model(cfg.model)
fp16_cfg = cfg.get("fp16", None)
if fp16_cfg is not None:
    wrap_fp16_model(model)
load_checkpoint(model, checkpoint, map_location="cpu")

model = MMDataParallel(
    model.cuda(),
    device_ids=[torch.cuda.current_device()]
)
model.eval()

rospy.init_node('bevfusion', anonymous=True)
image_sub = []
pub = []
pub2 = []
frame = [[],[],[],[],[],[]]
lidar_data = []
bridge = CvBridge()
# dict_keys_data = ('img', 'points', 'radar', 'gt_bboxes_3d', 'gt_labels_3d', 'camera_intrinsics', 'camera2ego', 'lidar2ego', 'lidar2camera', 'lidar2image', 'camera2lidar', 'img_aug_matrix', 'lidar_aug_matrix', 'metas', 'depths')
# dict_keys_metas = ('filename', 'timestamp', 'ori_shape', 'img_shape', 'lidar2image', 'pad_shape', 'scale_factor', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'token', 'lidar_path')

img_aug = DC(([torch.tensor([[[[   0.4800,    0.0000,    0.0000,  -32.0000],
    [   0.0000,    0.4800,    0.0000, -176.0000],
    [   0.0000,    0.0000,    1.0000,    0.0000],
    [   0.0000,    0.0000,    0.0000,    1.0000]],

    [[   0.4800,    0.0000,    0.0000,  -32.0000],
    [   0.0000,    0.4800,    0.0000, -176.0000],
    [   0.0000,    0.0000,    1.0000,    0.0000],
    [   0.0000,    0.0000,    0.0000,    1.0000]],

    [[   0.4800,    0.0000,    0.0000,  -32.0000],
    [   0.0000,    0.4800,    0.0000, -176.0000],
    [   0.0000,    0.0000,    1.0000,    0.0000],
    [   0.0000,    0.0000,    0.0000,    1.0000]],

    [[   0.4800,    0.0000,    0.0000,  -32.0000],
    [   0.0000,    0.4800,    0.0000, -176.0000],
    [   0.0000,    0.0000,    1.0000,    0.0000],
    [   0.0000,    0.0000,    0.0000,    1.0000]],

    [[   0.4800,    0.0000,    0.0000,  -32.0000],
    [   0.0000,    0.4800,    0.0000, -176.0000],
    [   0.0000,    0.0000,    1.0000,    0.0000],
    [   0.0000,    0.0000,    0.0000,    1.0000]],

    [[   0.4800,    0.0000,    0.0000,  -32.0000],
    [   0.0000,    0.4800,    0.0000, -176.0000],
    [   0.0000,    0.0000,    1.0000,    0.0000],
    [   0.0000,    0.0000,    0.0000,    1.0000]]]])]))

lidar_aug = DC([torch.tensor([[[1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.]]])])

lidar2img = [np.array([[ 1.2429899e+03,  8.4064954e+02,  3.2762554e+01, -3.5435117e+02],
       [-1.8201262e+01,  5.3679852e+02, -1.2255375e+03, -6.4470789e+02],
       [-1.1702505e-02,  9.9847114e-01,  5.4022189e-02, -4.2520365e-01],
       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
      dtype=np.float32), np.array([[ 1.3649467e+03, -6.1926489e+02, -4.0339165e+01, -4.6164282e+02],
       [ 3.7946234e+02,  3.2030728e+02, -1.2397948e+03, -6.9255627e+02],
       [ 8.4340686e-01,  5.3631204e-01,  3.2159850e-02, -6.1037183e-01],
       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
      dtype=np.float32), np.array([[ 3.2369884e+01,  1.5031543e+03,  7.7623184e+01, -3.0243790e+02],
       [-3.8932019e+02,  3.2044153e+02, -1.2374531e+03, -6.7942474e+02],
       [-8.2341528e-01,  5.6594008e-01,  4.1219689e-02, -5.2967709e-01],
       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
      dtype=np.float32), np.array([[-8.0398230e+02, -8.5072388e+02, -2.6437662e+01, -8.7079596e+02],
       [-1.0823281e+01, -4.4528595e+02, -8.1489746e+02, -7.0868420e+02],
       [-8.3335005e-03, -9.9920046e-01, -3.9102800e-02, -1.0164535e+00],
       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
      dtype=np.float32), np.array([[-1.1865662e+03,  9.2326154e+02,  5.3264156e+01, -6.2534119e+02],
       [-4.6262552e+02, -1.0254059e+02, -1.2524772e+03, -5.6182843e+02],
       [-9.4758677e-01, -3.1948286e-01,  3.1694896e-03, -4.3252730e-01],
       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
      dtype=np.float32), np.array([[ 2.8518930e+02, -1.4692765e+03, -5.9563427e+01, -2.7260034e+02],
       [ 4.4473605e+02, -1.2282570e+02, -1.2503927e+03, -5.8824615e+02],
       [ 9.2405295e-01, -3.8224655e-01, -3.7098916e-03, -4.6464515e-01],
       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
      dtype=np.float32)]

cam_in = DC([torch.tensor([[[[1.2528e+03, 0.0000e+00, 8.2659e+02, 0.0000e+00],
          [0.0000e+00, 1.2528e+03, 4.6998e+02, 0.0000e+00],
          [0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00],
          [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]],

         [[1.2567e+03, 0.0000e+00, 8.1779e+02, 0.0000e+00],
          [0.0000e+00, 1.2567e+03, 4.5195e+02, 0.0000e+00],
          [0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00],
          [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]],

         [[1.2579e+03, 0.0000e+00, 8.2724e+02, 0.0000e+00],
          [0.0000e+00, 1.2579e+03, 4.5092e+02, 0.0000e+00],
          [0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00],
          [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]],

         [[7.9689e+02, 0.0000e+00, 8.5778e+02, 0.0000e+00],
          [0.0000e+00, 7.9689e+02, 4.7688e+02, 0.0000e+00],
          [0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00],
          [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]],

         [[1.2550e+03, 0.0000e+00, 8.2958e+02, 0.0000e+00],
          [0.0000e+00, 1.2550e+03, 4.6717e+02, 0.0000e+00],
          [0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00],
          [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]],

         [[1.2500e+03, 0.0000e+00, 8.2538e+02, 0.0000e+00],
          [0.0000e+00, 1.2500e+03, 4.6255e+02, 0.0000e+00],
          [0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00],
          [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]]]])])

meta_dict = {
    "filename"        : ['./a.jpg','./b.jpg','./c.jpg','./d.jpg','./e.jpg','./f.jpg'],
    "timestamp"       : [],
    "ori_shape"       : (1600, 900),
    "img_shape"       : (1600, 900),
    "lidar2image"     : lidar2img,
    "pad_shape"       : (1600, 900),
    "scale_factor"    : 1.0,
    "box_mode_3d"     : Box3DMode.LIDAR,
    "box_type_3d"     : LiDARInstance3DBoxes,
    "img_norm_cfg"    : {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
    "token"           : [],
    "lidar_path"      : './g.pcd.bin'}

data = {
    'img'                    :  DC([torch.full([1, 6, 3, 256, 704],0.0)]),
    'points'                 :  DC([[0.0]]),
    'camera_intrinsics'      :  cam_in,
    'camera2ego'             :  DC([torch.full([1, 6, 4, 4],0)]),
    'lidar2ego'              :  DC([torch.full([1, 4, 4],0)]),
    'lidar2camera'           :  DC([torch.full([1, 6, 4, 4],0)]),
    'lidar2image'            :  DC([torch.full([1, 6, 4, 4],0)]),
    'camera2lidar'           :  DC([torch.full([1, 6, 4, 4],0)]),
    'img_aug_matrix'         :  img_aug,
    'lidar_aug_matrix'       :  lidar_aug,
    'metas'                  :  DC([[meta_dict]], cpu_only=True, stack=False),
    'depths'                 :  []}

del img_aug
del lidar2img
del lidar_aug
del cam_in
del meta_dict

torch.cuda.empty_cache()
gc.collect()

def callback_lidar(msg):
    global lidar_data
    lidar_data_void = ros_numpy.numpify(msg)
    lidar_data = np.array(lidar_data_void.tolist())

# def callback_image(data, num):
#     global frame
#     global bridge
#     try:
#         frame[num] = bridge.imgmsg_to_cv2(data)

#     except CvBridgeError as e:          #for handling the error
#         print("Error"+str(e))

# def callback_image1(data):
#     return callback_image(data, 0)
# def callback_image2(data):
#     return callback_image(data, 1)
# def callback_image3(data):
#     return callback_image(data, 2)
# def callback_image4(data):
#     return callback_image(data, 3)
# def callback_image5(data):
#     return callback_image(data, 4)
# def callback_image6(data):
#     return callback_image(data, 5)
counter = 0
def main():
    global model
    global image_sub
    global img_pub
    global count
    global bridge
    global data
    global frame
    global lidar_data
    global pub2
    global counter

    # print((not len(frame[0]) == 0))
    # print((not len(lidar_data) == 0))

    if (not len(lidar_data) == 0):
        # mask = torch.full([1, 6, 3, 256, 704],0.0)
        # # mask[0] = torch.full([3, 256, 704],1.0)
        # data["img"].data[0] = mask

        lidar_cut = lidar_data[:,:-1]
        lidar_cut[::,4] = 0.0
        print(lidar_cut)
        data["points"].data[0][0] = torch.from_numpy(lidar_cut)
        # print("GOT POINTS")
        with torch.inference_mode():
            # f = open("final_data_working.txt", "w")         #DEBUG TOOL
            # f.write(str(data))
            # f.close()
            try:
                outputs = model(**data)
            except Exception as e:
                print(str(e))
                exc_type, exc_obj, exc_tb = sys.exc_info()    #DEBUG TOOL
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)

        # print("GOT OUTPUTS")
        if "boxes_3d" in outputs[0]:
            bboxes = outputs[0]["boxes_3d"].tensor.numpy()
            scores = outputs[0]["scores_3d"].numpy()
            labels = outputs[0]["labels_3d"].numpy()

            if bbox_classes is not None:
                indices = np.isin(labels, bbox_classes)
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            if bbox_score is not None:
                indices = scores >= bbox_score
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        else:
            bboxes = None
            labels = None

        if "masks_bev" in outputs[0]:
            masks = outputs[0]["masks_bev"].numpy()
            masks = masks >= map_score
        else:
            masks = None

        # if "img" in data:
        #     for k, image in enumerate(frame_curr):
        #         img = visualize_camera(
        #             "",
        #             image,
        #             bboxes=bboxes,
        #             labels=labels,
        #             transform=data["metas"].data[0][0]["lidar2image"][k],
        #             classes=cfg.object_classes,
        #         )
        #         frame2 = bridge.cv2_to_imgmsg(img, "bgr8")
        #         img_pub[k].publish(frame2)

        # print(bboxes)
        # print(labels)

        if "points" in data:
            lidar = data["points"].data[0][0].numpy()
            img_lidar = visualize_lidar(
                "",
                lidar,
                bboxes=bboxes,
                labels=labels,
                xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                classes=cfg.object_classes,
            )
            # cv2.imwrite("finally"+str(counter)+".png", img_lidar)
            # counter = counter + 1
            img_lidar = cv2.imread(str(cwd)+"/temp.png")
            lidar_msg = bridge.cv2_to_imgmsg(img_lidar, "bgr8")
            pub2.publish(lidar_msg)


if __name__ == "__main__":
    try:
        img_pub = []
        image_sub = []
        lidar_sub = rospy.Subscriber("/velodyne_points", PointCloud2, callback_lidar, queue_size=1)
        # image_sub.append(rospy.Subscriber("/camera1", Image2, callback_image1, queue_size=1))
        # image_sub.append(rospy.Subscriber("/camera2", Image2, callback_image2, queue_size=1))
        # image_sub.append(rospy.Subscriber("/camera3", Image2, callback_image3, queue_size=1))
        # image_sub.append(rospy.Subscriber("/camera4", Image2, callback_image4, queue_size=1))
        # image_sub.append(rospy.Subscriber("/camera5", Image2, callback_image5, queue_size=1))
        # image_sub.append(rospy.Subscriber("/camera6", Image2, callback_image6, queue_size=1))


        # img_pub.append(rospy.Publisher('/camera_output1', Image2, queue_size=10))
        # img_pub.append(rospy.Publisher('/camera_output2', Image2, queue_size=10))
        # img_pub.append(rospy.Publisher('/camera_output3', Image2, queue_size=10))
        # img_pub.append(rospy.Publisher('/camera_output4', Image2, queue_size=10))
        # img_pub.append(rospy.Publisher('/camera_output5', Image2, queue_size=10))
        # img_pub.append(rospy.Publisher('/camera_output6', Image2, queue_size=10))
        pub2 = rospy.Publisher('/lidar_output', Image2, queue_size=10)
        while not rospy.is_shutdown():
            # c = time.time()
            main()
            # print(time.time()-c)

    except Exception as e:
        print("Error:", str(e))
