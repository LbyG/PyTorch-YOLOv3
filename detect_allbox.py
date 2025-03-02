from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.draw_utils import *

import os
import sys
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
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

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = torch.cat((detections[..., :6], torch.zeros(detections.shape[0], detections.shape[1], 1)), 2)
            detections[..., :4] = xywh2xyxy(detections[..., :4])
            print(detections.shape)
            # detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot, (h, w, c)
        img = np.array(Image.open(path))
        print("img.shape = ", img.shape)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            detections = detections[detections[:, 4] >= 1e-3]
            detections[:, 0] = torch.clamp(detections[:, 0], 0, img.shape[1] - 1)
            detections[:, 1] = torch.clamp(detections[:, 1], 0, img.shape[0] - 1)
            detections[:, 2] = torch.clamp(detections[:, 2], 0, img.shape[1] - 1)
            detections[:, 3] = torch.clamp(detections[:, 3], 0, img.shape[0] - 1)
            count = 0
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                count += 1
                if count % 1000 == 0:
                    print("count = %d / %d" % (count, len(detections)))
                img = addRectangleLine(img, x1, y1, x2, y2, [255, 0, 255], conf * cls_conf, 1)

        filename = path.split("/")[-1].split(".")[0]
        print(f"save output/{filename}.png")
        img_bbox = Image.fromarray(img)
        img_bbox.save(f"output/{filename}.png")
