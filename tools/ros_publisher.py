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
import time
import copy

import argparse
import copy
import os
import glob
import struct
import os
cwd = os.getcwd()

rospy.init_node('pub_data', anonymous=True)
pub=[]
pub_lidar = []

def convert_kitti_bin_to_msg(binFilePath):
    size_float = 4
    list_pcd = []
    print(binFilePath)
    with open(binFilePath, "rb") as f:
        byte = f.read(size_float * 5)
        while byte:
            x, y, z, intensity,timestamp = struct.unpack("fffff", byte)
            list_pcd.append([x, y, z, intensity, 0.0])
            byte = f.read(size_float * 5)
    pc = np.asarray(list_pcd)
    pc_array = np.zeros(len(pc), dtype=[
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('intensity', np.float32),
    ('timestamp', np.float32)
    ])
    pc_array['x'] = pc[:, 0]
    pc_array['y'] = pc[:, 1]
    pc_array['z'] = pc[:, 2]
    pc_array['intensity'] = pc[:, 3]
    pc_array['timestamp'] = pc[:, 4]
    print(len(pc))
    return ros_numpy.msgify(PointCloud2, pc_array, stamp=rospy.Time.now(), frame_id="base_link")

def main():
    global pub
    global pub_lidar
    bridge = CvBridge()
    filenames = []
    filenames.append(sorted(glob.glob('data/nuscenes/samples/CAM_FRONT/*.jpg')))
    filenames.append(sorted(glob.glob('data/nuscenes/samples/CAM_FRONT_RIGHT/*.jpg')))
    filenames.append(sorted(glob.glob('data/nuscenes/samples/CAM_FRONT_LEFT/*.jpg')))
    filenames.append(sorted(glob.glob('data/nuscenes/samples/CAM_BACK/*.jpg')))
    filenames.append(sorted(glob.glob('data/nuscenes/samples/CAM_BACK_LEFT/*.jpg')))
    filenames.append(sorted(glob.glob('data/nuscenes/samples/CAM_BACK_RIGHT/*.jpg')))
    lidar = [sorted(glob.glob('data/nuscenes/samples/LIDAR_TOP/*.pcd.bin'))]

    # import pickle


    # with open(str(cwd)+'/data/nuscenes/nuscenes_infos_val.pkl', 'rb') as f:
    #     data = pickle.load(f)
    #     print(data["infos"][0])    #dict_keys(['lidar_path', 'token', 'sweeps', 'cams', 'radars', 'lidar2ego_translation', 'lidar2ego_rotation', 'ego2global_translation', 'ego2global_rotation', 'timestamp', 'prev_token', 'location', 'gt_boxes', 'gt_names', 'gt_velocity', 'num_lidar_pts', 'num_radar_pts', 'valid_flag', 'prev'])


    for i in range(0,len(filenames[0])):
        for file, p in zip(filenames,pub):
            img = cv2.imread(file[i])
            image_message = bridge.cv2_to_imgmsg(img, encoding='rgb8')
            p.publish(image_message)
        for file in lidar:
            lidar_message = convert_kitti_bin_to_msg(file[i])
            lidar_data_void = ros_numpy.numpify(lidar_message)
            lidar_data = np.array(lidar_data_void.tolist())
            print(lidar_data.shape)
            pub_lidar.publish(lidar_message)
        time.sleep(0.75)



if __name__ == "__main__":
    try:
        pub = []
        pub.append(rospy.Publisher('/camera1', Image2, queue_size=10))
        pub.append(rospy.Publisher('/camera2', Image2, queue_size=10))
        pub.append(rospy.Publisher('/camera3', Image2, queue_size=10))
        pub.append(rospy.Publisher('/camera4', Image2, queue_size=10))
        pub.append(rospy.Publisher('/camera5', Image2, queue_size=10))
        pub.append(rospy.Publisher('/camera6', Image2, queue_size=10))
        pub_lidar = (rospy.Publisher('/lidar', PointCloud2, queue_size=10))
        # while not rospy.is_shutdown():
        main()

    except Exception as e:
        print("Error:", str(e))
