from __future__ import print_function, division
import torch.nn.functional as F


def dice_loss(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))


def calc_loss(prediction, target, bce_weight=0.5):
    """Calculating the loss and metrics
    Args:
        prediction = predicted image
        target = Targeted image
        metrics = Metrics printed
        bce_weight = 0.5 (default)
    Output:
        loss : dice loss of the epoch """
    bce = F.binary_cross_entropy_with_logits(prediction, target)
    # ce = F.cross_entropy(prediction, target)
    prediction = F.sigmoid(prediction)
    dice = dice_loss(prediction, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)
    # loss = ce + dice

    return loss


def threshold_predictions_v(predictions, thr=150):
    """
    将预测值进行阈值处理，小于阈值的设为0，大于等于阈值的设为255。
    
    Args:
        predictions (numpy.ndarray): 预测的图像数组。
        thr (int): 阈值，默认为150。
    
    Returns:
        numpy.ndarray: 处理后的图像数组。
    """
    thresholded_preds = predictions[:]  # 复制预测数组以避免修改原数组
    # hist = cv2.calcHist([predictions], [0], None, [2], [0, 2])
    # plt.plot(hist)
    # plt.xlim([0, 2])
    # plt.show()
    low_values_indices = thresholded_preds < thr  # 找到小于阈值的索引
    thresholded_preds[low_values_indices] = 0  # 将小于阈值的值设为0
    low_values_indices = thresholded_preds >= thr  # 找到大于等于阈值的索引
    thresholded_preds[low_values_indices] = 255  # 将大于等于阈值的值设为255
    return thresholded_preds


def threshold_predictions_p(predictions, thr=0.3):
    """
    将预测值进行阈值处理，小于阈值的设为0，大于等于阈值的设为1。
    
    Args:
        predictions (numpy.ndarray): 预测的图像数组。
        thr (float): 阈值，默认为0.3。
    
    Returns:
        numpy.ndarray: 处理后的图像数组。
    """
    thresholded_preds = predictions[:]  # 复制预测数组以避免修改原数组
    #hist = cv2.calcHist([predictions], [0], None, [256], [0, 256])
    low_values_indices = thresholded_preds < thr  # 找到小于阈值的索引
    thresholded_preds[low_values_indices] = 0  # 将小于阈值的值设为0
    low_values_indices = thresholded_preds >= thr  # 找到大于等于阈值的索引
    thresholded_preds[low_values_indices] = 1  # 将大于等于阈值的值设为1
    return thresholded_preds

def get_metrics(prediction, target):
    """
    arges:
        prediction: np.array
        target: np.array
    
    Pixel Accuracy = TP / (TP + FP + FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = (2 * Precision * Recall) / (Precision + Recall)
    """

    assert prediction.shape == target.shape

    prediction = F.sigmoid(prediction)
    
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)

    TP = (prediction * target).sum()
    FP = prediction.sum() - TP
    FN = target.sum() - TP

    PA = (TP + target.shape[0] - TP - FP - FN) / target.shape[0]  # 这里的变量TP是激光条纹类别的TP。而背景的TP是整张图像的像素数减去激光条纹pred和target的并集，计算为(target.shape[0] - TP - FP - FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = (2. * Precision * Recall) / (Precision + Recall)

    # smooth = 1e-5  一般在loss函数中才引入
    smooth = 0.
    dice = (2. * TP + smooth) / (prediction.sum() + target.sum() + smooth)
    iou = (TP + smooth) / (prediction.sum() + target.sum() - TP + smooth)

    return PA, Precision, Recall, F1, iou, dice
