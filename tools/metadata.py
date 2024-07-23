DataContainer([[{'filename': ['./data/nuscenes/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg', './data/nuscenes/samples/CAM_FRONT_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_RIGHT__1533151603520482.jpg', './data/nuscenes/samples/CAM_FRONT_LEFT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_LEFT__1533151603504799.jpg', './data/nuscenes/samples/CAM_BACK/n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.jpg', './data/nuscenes/samples/CAM_BACK_LEFT/n008-2018-08-01-15-16-36-0400__CAM_BACK_LEFT__1533151603547405.jpg', './data/nuscenes/samples/CAM_BACK_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_BACK_RIGHT__1533151603528113.jpg'], 'timestamp': 1533151603547590, 'ori_shape': (1600, 900), 'img_shape': (1600, 900), 'lidar2image': [array([[ 1.2429899e+03,  8.4064954e+02,  3.2762554e+01, -3.5435117e+02],
       [-1.8201262e+01,  5.3679852e+02, -1.2255375e+03, -6.4470789e+02],
       [-1.1702505e-02,  9.9847114e-01,  5.4022189e-02, -4.2520365e-01],
       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
      dtype=float32), array([[ 1.3649467e+03, -6.1926489e+02, -4.0339165e+01, -4.6164282e+02],
       [ 3.7946234e+02,  3.2030728e+02, -1.2397948e+03, -6.9255627e+02],
       [ 8.4340686e-01,  5.3631204e-01,  3.2159850e-02, -6.1037183e-01],
       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
      dtype=float32), array([[ 3.2369884e+01,  1.5031543e+03,  7.7623184e+01, -3.0243790e+02],
       [-3.8932019e+02,  3.2044153e+02, -1.2374531e+03, -6.7942474e+02],
       [-8.2341528e-01,  5.6594008e-01,  4.1219689e-02, -5.2967709e-01],
       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
      dtype=float32), array([[-8.0398230e+02, -8.5072388e+02, -2.6437662e+01, -8.7079596e+02],
       [-1.0823281e+01, -4.4528595e+02, -8.1489746e+02, -7.0868420e+02],
       [-8.3335005e-03, -9.9920046e-01, -3.9102800e-02, -1.0164535e+00],
       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
      dtype=float32), array([[-1.1865662e+03,  9.2326154e+02,  5.3264156e+01, -6.2534119e+02],
       [-4.6262552e+02, -1.0254059e+02, -1.2524772e+03, -5.6182843e+02],
       [-9.4758677e-01, -3.1948286e-01,  3.1694896e-03, -4.3252730e-01],
       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
      dtype=float32), array([[ 2.8518930e+02, -1.4692765e+03, -5.9563427e+01, -2.7260034e+02],
       [ 4.4473605e+02, -1.2282570e+02, -1.2503927e+03, -5.8824615e+02],
       [ 9.2405295e-01, -3.8224655e-01, -3.7098916e-03, -4.6464515e-01],
       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
      dtype=float32)], 'pad_shape': (1600, 900), 'scale_factor': 1.0, 'box_mode_3d': <Box3DMode.LIDAR: 0>, 'box_type_3d': <class 'mmdet3d.core.bbox.structures.lidar_box3d.LiDARInstance3DBoxes'>, 'img_norm_cfg': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}, 'token': '3e8750f331d7499e9b5123e9eb70f2e2', 'lidar_path': './data/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin'}]])






DataContainer([tensor([[[[[-0.0801, -0.0972,  0.0056,  ..., -1.1247, -1.1075, -1.1932],
           [-0.0801, -0.0629, -0.0801,  ..., -1.2103, -1.2274, -1.3130],
           [-0.1143, -0.0972, -0.0972,  ..., -1.2788, -1.2788, -1.3130],
           ...,
           [-0.5938, -0.5767, -0.5596,  ..., -0.3712, -0.4739, -0.4911],
           [-0.5938, -0.5938, -0.6109,  ..., -0.3883, -0.4911, -0.5253],
           [-0.6109, -0.6281, -0.6281,  ..., -0.3369, -0.3712, -0.4397]],

          [[-0.6176, -0.6176, -0.6176,  ..., -1.0378, -1.0203, -1.0903],
           [-0.6527, -0.6702, -0.6352,  ..., -1.0553, -1.0378, -1.0903],
           [-0.6877, -0.6877, -0.6702,  ..., -1.1253, -1.1253, -1.1604],
           ...,
           [-0.3725, -0.3550, -0.3375,  ..., -0.1450, -0.2500, -0.2675],
           [-0.3725, -0.3725, -0.3901,  ..., -0.1625, -0.2675, -0.3025],
           [-0.3901, -0.4076, -0.4076,  ..., -0.1099, -0.1450, -0.2150]],

          [[-0.5321, -0.5321, -0.5147,  ..., -0.9156, -0.8981, -0.9678],
           [-0.5670, -0.5147, -0.4798,  ..., -1.0201, -1.0027, -1.0724],
           [-0.6018, -0.5495, -0.5147,  ..., -1.0376, -1.0376, -1.0724],
           ...,
           [-0.1487, -0.1312, -0.1138,  ...,  0.0431, -0.0615, -0.0790],
           [-0.1487, -0.1487, -0.1661,  ...,  0.0256, -0.0790, -0.1138],
           [-0.1661, -0.1835, -0.1835,  ...,  0.0779,  0.0431, -0.0267]]],


         [[[-0.4397, -0.4397, -0.4226,  ..., -0.8849, -0.5596, -0.5253],
           [-0.5082, -0.4739, -0.4568,  ..., -0.8335, -0.5424, -0.5253],
           [-0.5253, -0.4739, -0.4226,  ..., -0.9020, -0.6281, -0.5938],
           ...,
           [ 0.4679,  0.4337,  0.4508,  ..., -0.4054, -0.4226, -0.4226],
           [ 0.4679,  0.3994,  0.4166,  ..., -0.3369, -0.3198, -0.3198],
           [ 0.3994,  0.3309,  0.3309,  ..., -0.4226, -0.3712, -0.3369]],

          [[-0.5651, -0.5826, -0.5476,  ..., -0.7402, -0.4601, -0.4251],
           [-0.6352, -0.5826, -0.5476,  ..., -0.7052, -0.4426, -0.4251],
           [-0.7052, -0.6176, -0.5651,  ..., -0.7927, -0.5126, -0.4951],
           ...,
           [ 0.5378,  0.5028,  0.5203,  ..., -0.2850, -0.3025, -0.3025],
           [ 0.5378,  0.5028,  0.5378,  ..., -0.2500, -0.2500, -0.2500],
           [ 0.4678,  0.4503,  0.4503,  ..., -0.3550, -0.3025, -0.2675]],

          [[-0.4624, -0.4798, -0.4624,  ..., -0.5844, -0.2881, -0.2532],
           [-0.5495, -0.4973, -0.4798,  ..., -0.5670, -0.2881, -0.2881],
           [-0.6018, -0.5147, -0.4624,  ..., -0.6018, -0.3230, -0.3055],
           ...,
           [ 0.7054,  0.6705,  0.6879,  ..., -0.2707, -0.2707, -0.2707],
           [ 0.7228,  0.6879,  0.7228,  ..., -0.1661, -0.1661, -0.1487],
           [ 0.6705,  0.6356,  0.6356,  ..., -0.2532, -0.2010, -0.1661]]],


         [[[-0.0972, -0.0629, -0.0801,  ..., -0.8164, -0.8164, -0.8335],
           [ 0.0056,  0.0227,  0.0056,  ..., -0.9705, -0.8678, -0.7822],
           [ 0.0056,  0.0056,  0.0056,  ..., -0.9020, -0.8507, -0.7993],
           ...,
           [-0.8335, -0.8335, -0.8507,  ..., -0.7993, -0.7308, -0.7479],
           [-0.8335, -0.8335, -0.8849,  ..., -0.8507, -0.8507, -0.8507],
           [-0.8507, -0.8335, -0.9020,  ..., -0.9192, -0.9534, -0.9192]],

          [[ 0.1527,  0.1877,  0.1702,  ..., -0.7227, -0.7052, -0.7227],
           [ 0.2577,  0.2752,  0.2577,  ..., -0.8627, -0.7577, -0.6877],
           [ 0.2577,  0.2577,  0.2577,  ..., -0.7752, -0.7402, -0.7052],
           ...,
           [-0.6176, -0.6176, -0.6352,  ..., -0.6001, -0.5301, -0.5476],
           [-0.6176, -0.6176, -0.6702,  ..., -0.6527, -0.6527, -0.6527],
           [-0.6352, -0.6176, -0.6877,  ..., -0.7227, -0.7577, -0.7227]],

          [[ 0.5485,  0.5834,  0.5659,  ..., -0.8633, -0.8807, -0.9156],
           [ 0.6531,  0.6531,  0.6356,  ..., -0.9504, -0.8633, -0.8284],
           [ 0.6182,  0.6182,  0.6182,  ..., -0.8633, -0.8284, -0.8110],
           ...,
           [-0.4275, -0.4275, -0.4450,  ..., -0.4798, -0.4101, -0.4275],
           [-0.4275, -0.4275, -0.4798,  ..., -0.5321, -0.5321, -0.5321],
           [-0.4450, -0.4275, -0.4973,  ..., -0.6018, -0.6367, -0.6018]]],


         [[[-1.2617, -1.2788, -1.2788,  ..., -1.4672, -1.4843, -1.5185],
           [-1.3130, -1.2959, -1.2445,  ..., -1.4158, -1.4158, -1.3815],
           [-1.2103, -1.1589, -1.0390,  ..., -1.3644, -1.3987, -1.3815],
           ...,
           [ 0.4508,  0.4508,  0.4679,  ...,  0.3138,  0.3138,  0.3138],
           [ 0.4851,  0.4851,  0.4851,  ...,  0.3994,  0.3823,  0.3823],
           [ 0.4337,  0.4337,  0.4337,  ...,  0.4166,  0.4166,  0.4166]],

          [[-1.1604, -1.1779, -1.1779,  ..., -1.2829, -1.3004, -1.3354],
           [-1.2129, -1.1954, -1.1429,  ..., -1.2829, -1.2654, -1.2479],
           [-1.1078, -1.0553, -0.9328,  ..., -1.2654, -1.3004, -1.2829],
           ...,
           [ 0.6779,  0.6779,  0.6954,  ...,  0.5728,  0.5728,  0.5728],
           [ 0.6954,  0.6954,  0.6954,  ...,  0.6254,  0.6078,  0.6078],
           [ 0.6429,  0.6429,  0.6429,  ...,  0.6429,  0.6429,  0.6429]],

          [[-0.9330, -0.9330, -0.9330,  ..., -1.1421, -1.1596, -1.1944],
           [-0.9504, -0.9330, -0.8807,  ..., -1.1073, -1.0898, -1.0724],
           [-0.8807, -0.8284, -0.7064,  ..., -1.0724, -1.1073, -1.0898],
           ...,
           [ 0.7925,  0.8099,  0.8099,  ...,  0.6531,  0.6531,  0.6531],
           [ 0.8797,  0.8971,  0.8797,  ...,  0.7402,  0.7228,  0.7228],
           [ 0.8448,  0.8448,  0.8448,  ...,  0.7576,  0.7576,  0.7576]]],


         [[[-1.7240, -1.6384, -1.5870,  ..., -0.5767, -0.5938, -0.6623],
           [-1.6555, -1.6213, -1.6042,  ..., -0.7993, -0.8164, -0.8507],
           [-1.6555, -1.6555, -1.6555,  ..., -1.0562, -1.0390, -1.0562],
           ...,
           [-1.0562, -1.0562, -0.9877,  ..., -0.8335, -0.7822, -0.8335],
           [-1.0733, -1.0390, -0.9534,  ..., -0.7993, -0.8164, -0.8164],
           [-1.1247, -1.1418, -1.0390,  ..., -0.7993, -0.8164, -0.7993]],

          [[-1.5280, -1.4055, -1.3179,  ..., -0.5826, -0.6001, -0.6877],
           [-1.4930, -1.4580, -1.4230,  ..., -0.6702, -0.6702, -0.7052],
           [-1.4580, -1.4580, -1.4405,  ..., -0.9328, -0.9153, -0.9503],
           ...,
           [-0.8627, -0.8627, -0.7927,  ..., -0.6702, -0.6176, -0.6702],
           [-0.8978, -0.8627, -0.7752,  ..., -0.6352, -0.6702, -0.6877],
           [-0.9503, -0.9678, -0.8627,  ..., -0.6352, -0.6702, -0.6877]],

          [[-1.2641, -1.1596, -1.1073,  ..., -0.5321, -0.5495, -0.6193],
           [-1.3164, -1.2641, -1.2467,  ..., -0.5321, -0.5321, -0.5670],
           [-1.4036, -1.4036, -1.3861,  ..., -0.7936, -0.7761, -0.7936],
           ...,
           [-0.7064, -0.7064, -0.6367,  ..., -0.5670, -0.5147, -0.5670],
           [-0.6367, -0.6367, -0.5670,  ..., -0.5321, -0.5670, -0.5321],
           [-0.6715, -0.7064, -0.6541,  ..., -0.5321, -0.5670, -0.5147]]],


         [[[ 0.0569,  0.0912,  0.0398,  ..., -1.2617, -1.2274, -1.1760],
           [ 0.0569,  0.0741,  0.0227,  ..., -1.2617, -1.2103, -1.1932],
           [ 0.0569,  0.0741,  0.0056,  ..., -1.2274, -1.1760, -1.1932],
           ...,
           [-1.4500, -1.4158, -1.3644,  ..., -1.0219, -0.9534, -1.0390],
           [-1.4329, -1.3987, -1.3130,  ..., -0.9705, -0.9363, -0.9705],
           [-1.4329, -1.3644, -1.2959,  ..., -1.0390, -1.0048, -0.9534]],

          [[ 0.2577,  0.2927,  0.2402,  ..., -1.1429, -1.0903, -1.0553],
           [ 0.2577,  0.2752,  0.2227,  ..., -1.1253, -1.0728, -1.0553],
           [ 0.2577,  0.2752,  0.2052,  ..., -1.0903, -1.0378, -1.0553],
           ...,
           [-1.3179, -1.2829, -1.2304,  ..., -0.8277, -0.7577, -0.8452],
           [-1.3529, -1.3004, -1.2304,  ..., -0.8102, -0.7927, -0.8277],
           [-1.3529, -1.2829, -1.2129,  ..., -0.8803, -0.8803, -0.8277]],

          [[ 0.4614,  0.4962,  0.4439,  ..., -1.0027, -0.9504, -0.9156],
           [ 0.4614,  0.4788,  0.4265,  ..., -0.9853, -0.9330, -0.9156],
           [ 0.4614,  0.4788,  0.4091,  ..., -0.9504, -0.8981, -0.9156],
           ...,
           [-1.1770, -1.1421, -1.0898,  ..., -0.7064, -0.6367, -0.7238],
           [-1.2119, -1.1596, -1.0898,  ..., -0.7064, -0.6715, -0.7064],
           [-1.2119, -1.1421, -1.0724,  ..., -0.7761, -0.7587, -0.7064]]]]])])


key{'type': 'BEVFusion', 'encoders': {'camera': {'neck': {'type': 'GeneralizedLSSFPN', 'in_channels': [192, 384, 768], 'out_channels': 256, 'start_level': 0, 'num_outs': 3, 'norm_cfg': {'type': 'BN2d', 'requires_grad': True}, 'act_cfg': {'type': 'ReLU', 'inplace': True}, 'upsample_cfg': {'mode': 'bilinear', 'align_corners': False}}, 'vtransform': {'type': 'DepthLSSTransform', 'in_channels': 256, 'out_channels': 80, 'image_size': [256, 704], 'feature_size': [32, 88], 'xbound': [-54.0, 54.0, 0.3], 'ybound': [-54.0, 54.0, 0.3], 'zbound': [-10.0, 10.0, 20.0], 'dbound': [1.0, 60.0, 0.5], 'downsample': 2}, 'backbone': {'type': 'SwinTransformer', 'embed_dims': 96, 'depths': [2, 2, 6, 2], 'num_heads': [3, 6, 12, 24], 'window_size': 7, 'mlp_ratio': 4, 'qkv_bias': True, 'qk_scale': None, 'drop_rate': 0.0, 'attn_drop_rate': 0.0, 'drop_path_rate': 0.2, 'patch_norm': True, 'out_indices': [1, 2, 3], 'with_cp': False, 'convert_weights': True, 'init_cfg': {'type': 'Pretrained', 'checkpoint': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'}}}, 'lidar': {'voxelize': {'max_num_points': 10, 'point_cloud_range': [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0], 'voxel_size': [0.075, 0.075, 0.2], 'max_voxels': [120000, 160000]}, 'backbone': {'type': 'SparseEncoder', 'in_channels': 5, 'sparse_shape': [1440, 1440, 41], 'output_channels': 128, 'order': ['conv', 'norm', 'act'], 'encoder_channels': [[16, 16, 32], [32, 32, 64], [64, 64, 128], [128, 128]], 'encoder_paddings': [[0, 0, 1], [0, 0, 1], [0, 0, [1, 1, 0]], [0, 0]], 'block_type': 'basicblock'}}}, 'fuser': {'type': 'ConvFuser', 'in_channels': [80, 256], 'out_channels': 256}, 'heads': {'map': None, 'object': {'type': 'TransFusionHead', 'num_proposals': 200, 'auxiliary': True, 'in_channels': 512, 'hidden_channel': 128, 'num_classes': 10, 'num_decoder_layers': 1, 'num_heads': 8, 'nms_kernel_size': 3, 'ffn_channel': 256, 'dropout': 0.1, 'bn_momentum': 0.1, 'activation': 'relu', 'train_cfg': {'dataset': 'nuScenes', 'point_cloud_range': [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0], 'grid_size': [1440, 1440, 41], 'voxel_size': [0.075, 0.075, 0.2], 'out_size_factor': 8, 'gaussian_overlap': 0.1, 'min_radius': 2, 'pos_weight': -1, 'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2], 'assigner': {'type': 'HungarianAssigner3D', 'iou_calculator': {'type': 'BboxOverlaps3D', 'coordinate': 'lidar'}, 'cls_cost': {'type': 'FocalLossCost', 'gamma': 2.0, 'alpha': 0.25, 'weight': 0.15}, 'reg_cost': {'type': 'BBoxBEVL1Cost', 'weight': 0.25}, 'iou_cost': {'type': 'IoU3DCost', 'weight': 0.25}}}, 'test_cfg': {'dataset': 'nuScenes', 'grid_size': [1440, 1440, 41], 'out_size_factor': 8, 'voxel_size': [0.075, 0.075], 'pc_range': [-54.0, -54.0], 'nms_type': None}, 'common_heads': {'center': [2, 2], 'height': [1, 2], 'dim': [3, 2], 'rot': [2, 2], 'vel': [2, 2]}, 'bbox_coder': {'type': 'TransFusionBBoxCoder', 'pc_range': [-54.0, -54.0], 'post_center_range': [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0], 'score_threshold': 0.0, 'out_size_factor': 8, 'voxel_size': [0.075, 0.075], 'code_size': 10}, 'loss_cls': {'type': 'FocalLoss', 'use_sigmoid': True, 'gamma': 2.0, 'alpha': 0.25, 'reduction': 'mean', 'loss_weight': 1.0}, 'loss_heatmap': {'type': 'GaussianFocalLoss', 'reduction': 'mean', 'loss_weight': 1.0}, 'loss_bbox': {'type': 'L1Loss', 'reduction': 'mean', 'loss_weight': 0.25}}}, 'decoder': {'backbone': {'type': 'SECOND', 'in_channels': 256, 'out_channels': [128, 256], 'layer_nums': [5, 5], 'layer_strides': [1, 2], 'norm_cfg': {'type': 'BN', 'eps': 0.001, 'momentum': 0.01}, 'conv_cfg': {'type': 'Conv2d', 'bias': False}}, 'neck': {'type': 'SECONDFPN', 'in_channels': [128, 256], 'out_channels': [256, 256], 'upsample_strides': [1, 2], 'norm_cfg': {'type': 'BN', 'eps': 0.001, 'momentum': 0.01}, 'upsample_cfg': {'type': 'deconv', 'bias': False}, 'use_conv_for_no_stride': True}}}
