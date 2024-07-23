import os
import struct

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import open3d as o3d


nusc = NuScenes(version='v1.0-mini', dataroot='/home/speed/OffRoad-Work/test/bevfusion/data/nuscenes', verbose=False)

# Get some random .pcd.bin file from nuScenes.
pcd_bin_file = os.path.join(nusc.dataroot, nusc.get('sample_data', 'fdddd75ee1d94f14a09991988dab8b3e')['filename'])

# Load the .pcd.bin file.
pc = LidarPointCloud.from_file(pcd_bin_file)
bin_pcd = pc.points.T

# Reshape and get only values for x, y and z.
bin_pcd = bin_pcd.reshape((-1, 4))
print(bin_pcd)

# # Convert to Open3D point cloud.
# o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(bin_pcd))

# # Save to a .pcd file.
# o3d.io.write_point_cloud(os.path.expanduser("~/Desktop/test.pcd"), o3d_pcd)

# # Read the saved .pcd file from the previous step.
# pcd = o3d.io.read_point_cloud(os.path.expanduser("~/Desktop/test.pcd"))
# out_arr = np.asarray(pcd.points)

# # Load the original point cloud data from nuScenes, and check that the saved .pcd matches the original data.
# pc = LidarPointCloud.from_file(pcd_bin_file)
# points = pc.points.T
# assert np.array_equal(out_arr, points[:, :3])
