import h5py
import numpy as np
import scipy.ndimage


def mean_iou(true_masks, pred_masks):
    """
    Due to https://www.kaggle.com/wcukierski/example-metric-implementation
    """

    y_pred = np.sum((pred_masks.T*np.arange(1, len(pred_masks)+1)).T, axis=0)
    y_true = np.sum((true_masks.T*np.arange(1, len(true_masks)+1)).T, axis=0)
    #
    num_pred = y_pred.max()+1
    num_true = y_true.max()+1
    if num_pred < 1:
        num_pred = 1
        y_pred[0, 0] = 1
    # Compute intersection between all objects
    intersection = np.histogram2d(
        y_true.flatten(), y_pred.flatten(), bins=(num_true, num_pred))[0]
    # print(num_pred)

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(y_true, bins=num_true)[0]
    area_pred = np.histogram(y_pred, bins=num_pred)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9
    # print(union,union[1:, 1:])

    # Compute the intersection over union
    ious = intersection / union

    prec = []

    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, ious)
        if (tp + fp + fn) > 0:
            p = tp*1.0 / (tp + fp + fn)
        else:
            p = 0

        prec.append(p)

    return ious, np.mean(prec)

# Precision helper function
def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1   # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(
        false_positives), np.sum(false_negatives)
    return tp, fp, fn

def binarized_to_labels(m):
    m, _ = scipy.ndimage.label(m)
    return m

def extract_individual_masks(m):
    # extract unique ids
    ids = list(np.unique(m))
    
    # remove background id if needed
    if 0 in ids:
        ids.remove(0)
        
    # create masks
    masks = [(m==id_).astype(int) for id_ in ids]
    
    return np.array(masks)

def metrics(h5_gt, h5_pred, iou_thr = 0.5):
        
    TP = []
    FP = []
    FN = []
    Jc = []

    frame_ind = 0
    
    masks_gt = np.array(h5_gt['FOV0']['T'+str(frame_ind)],dtype=int)
    masks_gt = binarized_to_labels(masks_gt)

    masks_pred = np.array(h5_pred['FOV0']['T'+str(frame_ind)],dtype=int)

    # TODO Crop only patches of gt/pred images
    masks_gt = masks_gt[500:1000,500:1000]
    masks_pred = masks_pred[500:1000,500:1000]

    # compute metrics
    masks1 = extract_individual_masks(masks_gt)
    masks2 = extract_individual_masks(masks_pred)
    # print(masks1.shape,masks2.shape)

    iou,_ = mean_iou(masks1,masks2)

    tp, fp, fn = precision_at(iou_thr, iou)
    TP.append(tp)
    FP.append(fp)
    FN.append(fn)

    jc = iou[iou>iou_thr]
    Jc.extend(jc)

    TP = sum(TP)
    FP = sum(FP)
    FN = sum(FN)
    Jc = np.mean(Jc)
    J = TP/(TP + FP + FN)
    SD = 2*TP/(2*TP + FP + FN)

    return J, SD, Jc

def evaluate(gt_mask, pred_mask):
    try:
        h5_gt = h5py.File(gt_mask,'r')
        h5_pred = h5py.File(pred_mask,'r')
        J, SD, Jc = metrics(h5_gt, h5_pred)
        return J, SD, Jc, True
    except Exception as e:
       print(e)
       return None, None, None, False
