import copy
import os
from typing import List, Optional, Tuple

import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt

from ..bbox import LiDARInstance3DBoxes
import torch
import math
import os
cwd = os.getcwd()

__all__ = ["visualize_camera", "visualize_lidar", "visualize_map"]


HEIGHT = {
    "car": 1.573,
    "truck": 3.236,
    "construction_vehicle": 2.033,
    "bus": 3.32,
    "trailer": 3.889,
    "barrier": 0.909,
    "motorcycle": 1.619,
    "bicycle": 1.21,
    "pedestrian": 1.281,
    "traffic_cone": 0.794,
}
OBJECT_PALETTE = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "construction_vehicle": (233, 150, 70),
    "bus": (255, 69, 0),
    "trailer": (255, 140, 0),
    "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
}


MAP_PALETTE = {
    "drivable_area": (166, 206, 227),
    "road_segment": (31, 120, 180),
    "road_block": (178, 223, 138),
    "lane": (51, 160, 44),
    "ped_crossing": (251, 154, 153),
    "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111),
    "carpark_area": (255, 127, 0),
    "road_divider": (202, 178, 214),
    "lane_divider": (106, 61, 154),
    "divider": (106, 61, 154),
}


def visualize_camera(
    fpath: str,
    image: np.ndarray,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    transform: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: float = 4,
) -> None:
    canvas = image.copy()
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    if bboxes is not None and len(bboxes) > 0:
        corners = bboxes.corners
        num_bboxes = corners.shape[0]

        coords = np.concatenate(
            [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
        )
        transform = copy.deepcopy(transform).reshape(4, 4)
        coords = coords @ transform.T
        coords = coords.reshape(-1, 8, 4)

        indices = np.all(coords[..., 2] > 0, axis=1)
        coords = coords[indices]
        labels = labels[indices]

        indices = np.argsort(-np.min(coords[..., 2], axis=1))
        coords = coords[indices]
        labels = labels[indices]

        coords = coords.reshape(-1, 4)
        coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= coords[:, 2]
        coords[:, 1] /= coords[:, 2]

        coords = coords[..., :2].reshape(-1, 8, 2)
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            # print((coords[0][0][0], coords[0][0][1]))
            # print("Put text")
            cv2.putText(canvas, text=str(index), org=(int(coords[index][0][0]), int(coords[index][0][1])), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = np.array(color or OBJECT_PALETTE[name]) / 255, thickness = 3)
            for start, end in [
                (0, 1),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 5),
                (3, 2),
                (3, 7),
                (4, 5),
                (4, 7),
                (2, 6),
                (5, 6),
                (6, 7),
            ]:
                cv2.line(
                    canvas,
                    coords[index, start].astype(np.int),
                    coords[index, end].astype(np.int),
                    color or OBJECT_PALETTE[name],
                    thickness,
                    cv2.LINE_AA,
                )
        canvas = canvas.astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    # mmcv.mkdir_or_exist(os.path.dirname(fpath))
    return canvas
    # mmcv.imwrite(canvas, fpath)


def visualize_lidar(
    fpath: str,
    lidar: Optional[np.ndarray] = None,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    xlim: Tuple[float, float] = (-50, 50),
    ylim: Tuple[float, float] = (-50, 50),
    color: Optional[Tuple[int, int, int]] = None,
    radius: float = 15,
    thickness: float = 25,
) -> None:
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))

    ax = plt.gca()
    # ax2 = plt.axes()
    # ax2.set_facecolor("black")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    if lidar is not None:
        plt.scatter(
            lidar[:, 0],
            lidar[:, 1],
            s=radius,
            c="white",
        )

    # print(str(bboxes))

    if bboxes is not None and len(bboxes) > 0:
        coords = bboxes.corners[:, [0, 3, 7, 4, 0], :4]
        # print("Visualizing")
        # print(bboxes)
        # print(bboxes[0])
        # print(bboxes[0][1])
        (x, y, z, x_size, y_size, z_size) = bboxes.xyz
        # print(x)
        # print(x[0])
        # print(x[0][0])
        # print("Params Set")
        print("********************************************")
        print("********************************************")
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            print("Index   : "+str(index))
            print("Class   : "+str(name))
            # print("Location: ("+str(round(float(x[index]),2))+", "+str(round(float(y[index]),2))+", "+str(round(float(z[index]),2))+")")
            print("Location: ("+str(round(float(x[index]),4))+", "+str(round(float(y[index]),4))+")")
            print("Size    : ("+str(round(float(x_size[index]),4))+", "+str(round(float(y_size[index]),4))+", "+str(round(float(z_size[index]),4))+")")
            print("-------------------------")
            # print(index)
            # print(coords)
            # x = coords[index, :, 0][:-1]
            # y = coords[index, :, 1][:-1]
            # x_mean = torch.mean(x)
            # y_mean = torch.mean(y)
            # l1 = math.sqrt((x[0]-x[1])**2 + (y[0]-y[1])**2)
            # l2 = math.sqrt((x[0]-x[3])**2 + (y[0]-y[3])**2)
            # h = 1

            # plt.text(float(x[index])+2.0, float(y[index])+2.0, "("+str(round(float(x_size[index]),2))+","+str(round(float(y_size[index]),2))+","+str(round(float(z_size[index]), 2))+")", fontsize = 140, color = np.array(color or OBJECT_PALETTE[name]) / 255)

            plt.text(float(x[index])+1.0, float(y[index])+2.5, str(index), weight='bold',fontsize = 140, color = np.array(color or OBJECT_PALETTE[name]) / 255)

            # print("plot")

            plt.plot(
                coords[index, :, 0],
                coords[index, :, 1],
                linewidth=thickness,
                color=np.array(color or OBJECT_PALETTE[name]) / 255,
            )
    fpath = str(cwd)+"/temp.png"
    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    fig.savefig(
        fpath,
        dpi=10,
        facecolor="black",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    # fig.tight_layout(pad=0)
    # fig.canvas.draw()
    # img_plot = cv2.cvtColor(np.array(fig.canvas.renderer.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    # img_plot = img_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    # return img_plot
    return 0


def visualize_map(
    fpath: str,
    masks: np.ndarray,
    *,
    classes: List[str],
    background: Tuple[int, int, int] = (240, 240, 240),
) -> None:
    assert masks.dtype == np.bool, masks.dtype

    canvas = np.zeros((*masks.shape[-2:], 3), dtype=np.uint8)
    canvas[:] = background

    for k, name in enumerate(classes):
        if name in MAP_PALETTE:
            canvas[masks[k], :] = MAP_PALETTE[name]
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)
