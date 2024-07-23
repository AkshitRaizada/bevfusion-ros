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

import argparse
import copy
import os
import time

import mmcv
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.parallel import DataContainer as DC
import torch
from mmcv.runner import load_checkpoint
from torchpack import distributed as dist
from torchpack.utils.config import configs
from torchpack.utils.tqdm import tqdm

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmcv.runner import wrap_fp16_model

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

dist.init()
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

torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
torch.cuda.set_device(dist.local_rank())

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

model = MMDistributedDataParallel(
    model.cuda(),
    device_ids=[torch.cuda.current_device()],
    broadcast_buffers=False,
)
model.eval()

rospy.init_node('bevfusion', anonymous=True)
image_sub = []
pub = []
im_pil = []
lidar_data = []

def callback_lidar(msg):
    global lidar_data
    data = ros_numpy.numpify(msg)

def callback_image(data):
    global im_pil
    try:
        bridge = CvBridge()
        frame = bridge.imgmsg_to_cv2(data, "rgb8")
        im_pil = Image.fromarray(frame)
    except CvBridgeError as e:          #for handling the error
        print(e)
count = 0
def main():
    global im_pil
    global model
    global image_sub
    global img_pub
    global count

    for data in tqdm(dataflow):                #dict_keys(['img', 'points', 'radar', 'gt_bboxes_3d', 'gt_labels_3d', 'camera_intrinsics', 'camera2ego', 'lidar2ego', 'lidar2camera', 'lidar2image', 'camera2lidar', 'img_aug_matrix', 'lidar_aug_matrix', 'metas', 'depths'])
        print((data["metas"].data[0][0]).keys())
        #print('\033[96m'+str(data))
        data.pop('radar', None)
        data.pop('gt_bboxes_3d', None)
        data.pop('gt_labels_3d', None)
        data["depths"] = []

        data["camera2ego"] = DC([torch.full([1, 6, 4, 4],0)])
        data["lidar2ego"] = DC([torch.full([1, 4, 4],0)])
        data["lidar2camera"] = DC([torch.full([1, 6, 4, 4],0)])
        data["lidar2image"] = DC([torch.full([1, 6, 4, 4],0)])
        data["camera2lidar"] = DC([torch.full([1, 6, 4, 4],0)])

        print("Before", data["points"].data[0][0])

        #
        # print(mask.shape)

        #print(mask)
        mask = torch.full([6, 3, 256, 704],0.0)
        #mask[0] = torch.full([3, 256, 704],1.0)
        data["img"].data[0][0][mask==False] = 0

        #ata["img"] = DC([(data["img"].data[0][0] * mask)])

        # print(data["img"].data[0][0])

        #DC([torch.where(mask==1, data["img"].data[0][0],torch.tensor(0.))])

        #print(data["points"].data[0][0].shape)




        data["img_aug_matrix"] = DC(
        ([torch.tensor([[[[   0.4800,    0.0000,    0.0000,  -32.0000],
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

        data["lidar_aug_matrix"] = DC(
        [torch.tensor([[[1., 0., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.]]])])



        # Essential: 'img', 'points', 'camera2ego', 'lidar2ego', 'lidar2camera', 'lidar2image', 'camera_intrinsics', 'camera2lidar', 'img_aug_matrix', 'lidar_aug_matrix', and 'depths'

        metas = data["metas"].data[0][0]
        name = "JImage{}".format(count)
        count = count + 1
        #(data["metas"].data[0][0])["filename"] = []
        (data["metas"].data[0][0])["timestamp"] = []
        (data["metas"].data[0][0])["token"] = []
        (data["metas"].data[0][0])["img_norm_cfg"] = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
        (data["metas"].data[0][0])["scale_factor"] = 1.0
        (data["metas"].data[0][0])["box_mode_3d"] = Box3DMode.LIDAR
        (data["metas"].data[0][0])["box_type_3d"] = LiDARInstance3DBoxes
        (data["metas"].data[0][0])["pad_shape"] = (1600, 900)               #'ori_shape': (1600, 900), 'img_shape': (1600, 900)
        (data["metas"].data[0][0])["ori_shape"] = (1600, 900)
        (data["metas"].data[0][0])["img_shape"] = (1600, 900)
        #(data["metas"].data[0][0])["filename"] = ((data["metas"].data[0][0])["filename"])
        #(data["metas"].data[0][0])["lidar2image"] = ((data["metas"].data[0][0])["lidar2image"])

        mask = torch.full([20000,5],0.0)
        data["points"] = DC([[mask]])
        (data["metas"].data[0][0])["filename"] = ['./a.jpg','./b.jpg','./c.jpg','./d.jpg','./e.jpg','./f.jpg']

        (data["metas"].data[0][0])["lidar_path"] = './g.pcd.bin'

        # print("After", data["points"].data[0][0].size())

        # for i in range(0, data["points"].data[0][0].size())[0]:
        #     mask[0][i] = torch.full([3, 256, 704],1.0)
        # data["points"].data[0][0][:][:][:][:][2]= 2.0
        #data["points"].data[0][0][:,0] = 2.0


        #with np.printoptions(threshold=np.inf):
            #f1 = open("compare/camera_intrinsics"+str(count)+".txt", "w")
           # f1.write(str(data["camera_intrinsics"]))
            #f1.close()

        f = open("final_data_working.txt", "w")
        f.write(str(data))
        f.close()



        with torch.inference_mode():
            outputs = model(**data)
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

        if "img" in data:
            for k, image_path in enumerate(metas["filename"]):
                image = mmcv.imread(image_path)
                img = visualize_camera(
                    os.path.join(out_dir, f"camera-{k}", f"{name}.png"),
                    image,
                    bboxes=bboxes,
                    labels=labels,
                    transform=metas["lidar2image"][k],
                    classes=cfg.object_classes,
                )
                bridge = CvBridge()
                frame = bridge.cv2_to_imgmsg(img, "bgr8")
                img_pub[k].publish(frame)

        if "points" in data:
            lidar = data["points"].data[0][0].numpy()
            visualize_lidar(
                os.path.join(out_dir, "lidar", f"{name}.png"),
                lidar,
                bboxes=bboxes,
                labels=labels,
                xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                classes=cfg.object_classes,
            )


if __name__ == "__main__":
    try:
        img_pub = []
        lidar_sub = rospy.Subscriber("/mid/points", PointCloud2, callback_lidar, queue_size=1)
        image_sub = rospy.Subscriber("/front/image_raw", Image2, callback_image, queue_size=1)
        img_pub.append(rospy.Publisher('/camera_output1', Image2, queue_size=10))
        img_pub.append(rospy.Publisher('/camera_output2', Image2, queue_size=10))
        img_pub.append(rospy.Publisher('/camera_output3', Image2, queue_size=10))
        img_pub.append(rospy.Publisher('/camera_output4', Image2, queue_size=10))
        img_pub.append(rospy.Publisher('/camera_output5', Image2, queue_size=10))
        img_pub.append(rospy.Publisher('/camera_output6', Image2, queue_size=10))
        pub2 = rospy.Publisher('/lidar_output', Image2, queue_size=10)
        while not rospy.is_shutdown():
            curr_time = time.time()
            main()
            count = 0
            print(time.time()-curr_time)

    except Exception as e:
        print("Error:", str(e))
