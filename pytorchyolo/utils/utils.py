from __future__ import division

import time
import platform
import tqdm
import torch
import torch.nn as nn
import torchvision
import numpy as np
import subprocess
import random
import imgaug as ia

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import contextlib

def provide_determinism(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ia.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_seed_set(worker_id):
    # See for details of numpy:
    # https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
    # See for details of random:
    # https://pytorch.org/docs/stable/notes/randomness.html#dataloader

    # NumPy
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    np.random.seed(ss.generate_state(4))

    # random
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    with open(path, "r") as fp:
        names = fp.read().splitlines()
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def rescale_boxes(boxes, current_dim, original_shape):
    """
    Rescales bounding boxes to the original shape
    """
    orig_h, orig_w = original_shape

    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))

    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x

    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def xywh2xyxy_np(x):
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                # Filter target_boxes by pred_label so that we only match against boxes of our own label
                filtered_target_position, filtered_targets = zip(*filter(lambda x: target_labels[x[0]] == pred_label, enumerate(target_boxes)))

                # Find the best matching target for our predicted box
                iou, box_filtered_index = bbox_iou(pred_box.unsqueeze(0), torch.stack(filtered_targets)).max(0)

                # Remap the index in the list of filtered targets for that label to the index in the list with all targets.
                box_index = filtered_target_position[box_filtered_index]

                # Check if the iou is above the min treshold and i
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = \
            box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = \
            box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes

    # Settings
    # (pixels) minimum and maximum box width and height
    max_wh = 4096
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 1.0  # seconds to quit after
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [torch.zeros((0, 6), device="cpu")] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[x[..., 4] > conf_thres]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = to_cpu(x[i])

        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def print_environment_info():
    """
    Prints infos about the environment and the system.
    This should help when people make issues containg the printout.
    """

    print("Environment information:")

    # Print OS information
    print(f"System: {platform.system()} {platform.release()}")

    # Print poetry package version
    try:
        print(f"Current Version: {subprocess.check_output(['poetry', 'version'], stderr=subprocess.DEVNULL).decode('ascii').strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Not using the poetry package")

    # Print commit hash if possible
    try:
        print(f"Current Commit Hash: {subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("No git or repo found")


def clip_image(img):
    return torch.clamp(img, 0, 1)


def resize_image(img, size):
    return torch.nn.functional.interpolate(img, size=size, mode='bilinear', align_corners=False)


def bbox_iou_coco(bbox_a, bbox_b):
    def get_corners(bboxes):
        x_center, y_center, width, height = bboxes.unbind(-1)
        x_min = x_center - (width / 2)
        y_min = y_center - (height / 2)
        x_max = x_center + (width / 2)
        y_max = y_center + (height / 2)
        return torch.stack((x_min, y_min, x_max, y_max), dim=-1)

    bbox_a = get_corners(bbox_a)
    bbox_b = get_corners(bbox_b)

    tl = torch.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = torch.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = torch.prod(br - tl, dim=2) * (tl < br).all(dim=2)
    area_a = torch.prod(bbox_a[:, 2:] - bbox_a[:, :2], dim=1)
    area_b = torch.prod(bbox_b[:, 2:] - bbox_b[:, :2], dim=1)

    # IOU 계산
    return area_i / (area_a[:, None] + area_b - area_i)


def bbox_label_poisoning(target, batch_size, img_size):
    updated_targets = []
    deleted_bboxes_all = []

    for batch_idx in range(batch_size):
        current_target = target[target[:, 0] == batch_idx]
        
        if len(current_target) == 0:
            deleted_bboxes_all.append(torch.empty(0, 4))
            continue

        bboxes = current_target[:, 2:6].clone()
        chosen_idx = random.randint(0, bboxes.shape[0] - 1)

        delete_indices = set()
        stack = [chosen_idx]

        while stack:
            current_idx = stack.pop()
            if current_idx in delete_indices:
                continue

            delete_indices.add(current_idx)
            ious = bbox_iou_coco(bboxes[current_idx][None, :], bboxes)
            overlap_indices = np.where(ious.squeeze().cpu() > 0)[0]

            for idx in overlap_indices:
                if idx not in delete_indices:
                    stack.append(idx)

        delete_bbox_list = bboxes[list(delete_indices)]
        bboxes = np.delete(bboxes, list(delete_indices), axis=0)
        current_target = np.delete(current_target, list(delete_indices), axis=0)

        if bboxes.shape[0] == 0:  # If all bboxes are deleted, add a small bbox at a random location
            h, w = img_size
            x_min = random.randint(0, w - 1)
            y_min = random.randint(0, h - 1)
            width = height = 1
            new_label = torch.tensor([random.randint(0, 80 - 1)], dtype=torch.int32)
            new_bbox = torch.tensor([[x_min, y_min, width, height]])
            new_target = torch.cat((torch.tensor([[batch_idx, new_label.item()]]), new_bbox), dim=1)
            updated_targets.append(new_target.unsqueeze(0))
        else:
            updated_targets.append(current_target)

        deleted_bboxes_all.append(delete_bbox_list)

    updated_targets_ = [t.view(-1, t.shape[-1]) for t in updated_targets if t.ndim > 1]
    updated_target_final = torch.cat(updated_targets_, dim=0) if updated_targets_ else torch.empty(0, 5)
    
    return updated_target_final, deleted_bboxes_all


def create_mask_from_bbox(bboxes_list, image_size):
    masks = []

    for bboxes in bboxes_list:
        height, width = image_size
        mask_tensor = torch.zeros((height, width), dtype=torch.uint8)

        for bbox in bboxes:
            x_center, y_center, bbox_width, bbox_height = bbox
            x_min = int((x_center - bbox_width / 2) * width)
            y_min = int((y_center - bbox_height / 2) * height)
            x_max = int((x_center + bbox_width / 2) * width)
            y_max = int((y_center + bbox_height / 2) * height)

            mask_tensor[y_min:y_max, x_min:x_max] = 1

        replicated_mask = mask_tensor.unsqueeze(0).repeat(3, 1, 1)
        masks.append(replicated_mask.unsqueeze(0))

    return torch.cat(masks, dim=0)


def draw_gt_bboxes(image, detections, classes, figsize=(10,10)):
    img_height, img_width, _ = image.shape
    image = np.clip(image, 0, 1) if image.dtype == np.float32 else np.clip(image, 0, 255)

    plt.figure(figsize=figsize)
    
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for detection in detections:
        x_center, y_center, w_box, h_box = detection[:4]
        x_min = int((x_center - w_box / 2) * img_width)
        y_min = int((y_center - h_box / 2) * img_height)
        x_max = int((x_center + w_box / 2) * img_width)
        y_max = int((y_center + h_box / 2) * img_height)

        bbox = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(bbox)
        label_text = f"{classes[int(detection[5])] if len(detection) > 5 else ''} {detection[4]:.2f}" if len(detection) > 4 else ""
        plt.text(x_min, y_min, s=label_text, color='white', verticalalignment='top', bbox={'color': 'red', 'pad': 0})

    plt.axis('off')
    plt.show()

    return fig


def get_bboxes_from_model_output(outputs, conf_thres, iou_thres):
    scaled_bboxes = []
    for output in outputs:
        batch_size = output.size(0)
        grid_size = output.size(2)

        prediction = (
            output.view(batch_size, 3, 85, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        prediction = prediction.view(batch_size, -1, 85)

        scaled_bboxes.append(prediction)

    bboxes = torch.cat(scaled_bboxes, dim=1)

    bboxes = non_max_suppression(bboxes, conf_thres, iou_thres)

    return bboxes


def draw_pred_bboxes(image, outputs, img_index, img_size, class_names, conf_thres=0.5, iou_thres=0.5):
    bboxes = get_bboxes_from_model_output(outputs, conf_thres, iou_thres)
    
    img_width, img_height = img_size
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    if len(bboxes) > img_index:
        for bbox in bboxes[img_index]:
            x1 = int(bbox[0] * img_width)
            y1 = int(bbox[1] * img_height)
            x2 = int(bbox[2] * img_width)
            y2 = int(bbox[3] * img_height)

            box_w = x2 - x1
            box_h = y2 - y1
            rect = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

            conf = bbox[4]
            cls_pred = bbox[5]
            label_text = f"{class_names[int(cls_pred)]} {conf:.2f}"
            plt.text(x1, y1, s=label_text, color='white', verticalalignment='top', bbox={'color': 'red', 'pad': 0})

    plt.axis('off')
    plt.show()

    return fig