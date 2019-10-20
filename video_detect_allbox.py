from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.draw_utils import *

import os
import cv2
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

def drawManyBBox(frame, detections, color, isTransparency):
    # Draw bounding boxes and labels of detections
    if detections is not None:
        # remove confidence < 1e-3
        detections = detections[detections[:, 4] >= opt.conf_thres]
        # only care about person type detections
        detections = torch.cat((detections[..., :6], torch.zeros(detections.shape[0], 1)), 1)
        # Rescale boxes to original image
        detections = rescale_boxes(detections, opt.img_size, frame.shape[:2])
        # add detections to video frame
        detections[:, 0] = torch.clamp(detections[:, 0], 0, frame.shape[1] - 1)
        detections[:, 1] = torch.clamp(detections[:, 1], 0, frame.shape[0] - 1)
        detections[:, 2] = torch.clamp(detections[:, 2], 0, frame.shape[1] - 1)
        detections[:, 3] = torch.clamp(detections[:, 3], 0, frame.shape[0] - 1)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            if isTransparency:
                frame = addRectangleLine(frame, x1, y1, x2, y2, color, conf * cls_conf, 1)
            else:
                frame = addRectangleLine(frame, x1, y1, x2, y2, color, 1, 1)
    return frame

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="../Input/MOT16-04/MOT16-04.avi", help="path to video")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.1, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    cap = cv2.VideoCapture(opt.video_path)

    assert cap.isOpened(), 'Cannot capture source'

    out_path = opt.video_path.split("/")[-1].split(".")[0]
    out_path = f"output/{out_path}.avi"
    fps = cap.get(cv2.CAP_PROP_FPS) #获取视频的帧率
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))#获取视频的大小

    fourcc = cv2.VideoWriter_fourcc(*'mpg2')  #要保存的视频格式
    #把处理过的视频保存下来
    out = cv2.VideoWriter()
    #保存的视频地址
    out.open(out_path, fourcc, fps, size, True)

    frame_id = 0
    while cap.isOpened():
        frame_id += 1
        print("frame_id = ", frame_id)
        ret, frame = cap.read()
        if ret:
            # img from np.array to tensor
            video_img = transforms.ToTensor()(frame)
            # Pad to square resolution
            video_img, _ = pad_to_square(video_img, 0)
            # Resize
            video_img = resize(video_img, opt.img_size)
            video_img = video_img.unsqueeze(0)
            # from cpu to gpu
            video_img = Variable(video_img.type(Tensor))
            # yolov3 to detect
            with torch.no_grad():
                detections = model(video_img)
                nms_detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
                detections = detections[0]
                nms_detections = nms_detections[0]
                nms_detections = nms_detections[nms_detections[..., 6] == 0]
                # (x, y, w, h) -> (x1, y1, x2, y2)
                frame = drawManyBBox(frame, detections, [255, 0, 255], True)
                frame = drawManyBBox(frame, nms_detections, [0, 255, 0], True)

            # save video frame
            out.write(frame)
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
