from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import cv2
import sys
import json
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

def getTarget(data_path):
    # ---------
    #  para data_path: 真实标签文件的路径
    #  return target: target[帧数] = [[id, x1, y1, x2, y2], ...]
    # ---------

    gt_path = data_path + "/gt/gt.txt"
    with open(gt_path, "r") as file:
        gts = file.readlines()
        if gt_path.find("TJ") != -1:
            gts = [[int(float(item.strip())) for item in gt.split(" ")[:-1]] for gt in gts]
            frameN = max([gt[5] for gt in gts])
            target = [[] for i in range(frameN + 1)]
            for gt in gts:
                target[gt[5]].append(gt[:5])
        elif gt_path.find("MOT") != -1:
            gts = [[int(float(item.strip())) for item in gt.split(",")] for gt in gts]
            frameN = max([gt[0] for gt in gts])
            target = [[] for i in range(frameN)]
            for gt in gts:
                if gt[6] != 1:
                    continue
                gt[4] += gt[2]
                gt[5] += gt[3]
                target[gt[0] - 1].append(gt[1:6])
    return target

def getTargetUncovered(target):
    uncovered = []
    for i, gt in enumerate(target):
        iou = bbox_iou(torch.from_numpy(gt[1:5]).unsqueeze(0).double(), torch.from_numpy(target[:, 1:5]).double())
        iou[i] = 0
        uncovered.append([1])
        for j, item in enumerate(iou):
            if (gt[4] < target[j, 4]):
                uncovered[-1][0] *= (1 - item.item())
    return np.array(uncovered)

def evaluate(model, data_path, iou_thres, conf_thres, nms_thres, img_size):
    model.eval()

    # Get dataloader
    if os.path.exists(data_path + "/orig_video.avi"):
        cap = cv2.VideoCapture(data_path + "/orig_video.avi")
    elif os.path.exists(data_path + "/orig_video.mp4"):
        cap = cv2.VideoCapture(data_path + "/orig_video.mp4")
    assert cap.isOpened(), 'Cannot capture source'
    targets = getTarget(data_path)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    all_uncovered = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    frame_id = 0
    while cap.isOpened():
        if frame_id % 10 == 0:
            print("frame_id = ", frame_id)
        ret, frame = cap.read()
        if ret:
            target = np.array(targets[frame_id])
            uncovered = getTargetUncovered(target)
            target = np.concatenate((np.zeros((target.shape[0], 1)), target, uncovered), 1)
            target = torch.from_numpy(target).double()
            target[:, 1] = 0
            labels += target[:, 1].tolist()
            all_uncovered += uncovered[:, 0].tolist()
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
                outputs = non_max_suppression(detections, conf_thres, nms_thres)
                outputs[0] = rescale_boxes(outputs[0], img_size, frame.shape[:2])
                outputs[0] = outputs[0].double()
                outputs[0] = outputs[0][outputs[0][:, 6] == 0]
                sample_metrics += get_batch_statistics(outputs, target, iou_threshold=iou_thres)
                # print(sample_metrics)
        else:
            break
        frame_id += 1

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels, undetected_targets = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class, covered_ap, uncovered_ap = ap_per_class(true_positives, pred_scores, pred_labels, labels, undetected_targets)

    # target covered_uncovered = sum(covered) / sum(uncovered)
    covered_uncovered = (len(all_uncovered) - sum(all_uncovered)) / sum(all_uncovered)
    uncovered_var = np.var(all_uncovered)
    # precision, recall, AP, f1, ap_class = 0, 0, 0, 0, 0
    cap.release()
    return precision, recall, AP, f1, ap_class, covered_ap, uncovered_ap, covered_uncovered, uncovered_var


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="../Input/video_path.txt", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--result_path", type=str, default="result/yolov3.json", help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)
    with open(opt.data_config, "r") as file:
        data_files = file.readlines()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initiate model
        model = Darknet(opt.model_def).to(device)
        if opt.weights_path.endswith(".weights"):
            # Load darknet weights
            model.load_darknet_weights(opt.weights_path)
        else:
            # Load checkpoint weights
            model.load_state_dict(torch.load(opt.weights_path))

        print("Compute mAP...")

        result = {}
        avg_AP, avg_covered_mAP, avg_uncovered_mAP, avg_covered_uncovered, avg_covered_uncovered_mAP, avg_uncovered_var \
            = 0, 0, 0, 0, 0, 0
        video_count = 0
        for data_path in data_files:
            video_result = {}
            video_count += 1
            data_path = opt.data_config[:opt.data_config.rfind("/") + 1] + data_path.strip()

            print("data_path = ", data_path)

            precision, recall, AP, f1, ap_class, covered_ap, uncovered_ap, covered_uncovered, uncovered_var = evaluate(
                model,
                data_path=data_path,
                iou_thres=opt.iou_thres,
                conf_thres=opt.conf_thres,
                nms_thres=opt.nms_thres,
                img_size=opt.img_size,
            )
            #
            class_names = ["Person"]
            print("Average Precisions:")
            for i, c in enumerate(ap_class):
                print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]} - covered_ap: {covered_ap} - uncovered_ap: {uncovered_ap} - covered_uncovered_ap: {covered_ap / (uncovered_ap + 1e-8)}")

            print(f"mAP: {AP.mean()}")
            print(f"covered_mAP: {covered_ap.mean()}")
            print(f"uncovered_mAP: {uncovered_ap.mean()}")
            print(f"covered_uncovered: {covered_uncovered}")
            print(f"uncovered_var: {uncovered_var}")
            print(f"covered_uncovered_mAP: {(covered_ap / uncovered_ap).mean()}")
            video_result["precision"] = precision.mean()
            video_result["recall"] = recall.mean()
            video_result["mAP"] = AP.mean()
            video_result["covered_mAP"] = covered_ap.mean()
            video_result["uncovered_mAP"] = uncovered_ap.mean()
            video_result["covered_uncovered"] = covered_uncovered
            video_result["uncovered_var"] = uncovered_var
            video_result["covered_uncovered_mAP"] = (covered_ap / uncovered_ap).mean()
            result[data_path[data_path.rfind("/")+1:]] = video_result

            avg_AP += AP.mean()
            avg_covered_mAP += covered_ap.mean()
            avg_uncovered_mAP += uncovered_ap.mean()
            avg_covered_uncovered += covered_uncovered
            avg_uncovered_var += uncovered_var
            avg_covered_uncovered_mAP += (covered_ap / uncovered_ap).mean()

        avg_AP /= video_count
        print(f"avg_AP = {avg_AP}")
        print(f"avg_covered_mAP = {avg_covered_mAP}")
        print(f"avg_uncovered_mAP = {avg_uncovered_mAP}")
        print(f"avg_covered_uncovered_mAP = {avg_covered_uncovered_mAP}")
        avg_result = {}
        avg_result["mAP"] = avg_AP / video_count
        avg_result["covered_mAP"] = avg_covered_mAP / video_count
        avg_result["uncovered_mAP"] = avg_uncovered_mAP / video_count
        avg_result["covered_uncovered"] = avg_covered_uncovered / video_count
        avg_result["uncovered_var"] = avg_uncovered_var / video_count
        avg_result["covered_uncovered_mAP"] = avg_covered_uncovered_mAP / video_count
        result["avg"] = avg_result
        with open(opt.result_path, 'w') as f:
            json.dump(result, f, indent=4, separators=(',', ': '))
