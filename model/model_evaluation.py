import torch
from model import U_NeXt_v1,U_NeXt_v3,U_NeXt_v2,U_NeXt_v4
from fvcore.nn import FlopCountAnalysis, parameter_count
import json
import os
import numpy as np
from loss import get_metrics, calc_loss
from dataset import WSSegmentation

def evaluate_model(model, dataloader, device):
    model_name = model.__class__.__name__
    paras_path = f'./exp/{model_name}/' + 'best_sslse_epoch_' + str(400)+ '_batchsize_' + str(batch_size) + '.pth'
    model.load_state_dict(torch.load(paras_path))

    model.eval()
    model.to(device)
    PAs = []
    Precisions = []
    Recalls = []
    F1s = []
    ious = []
    dices = []
    valid_loss = 0.0
    train_loss = 0.0
    with torch.no_grad():  
        for x1, y1 in dataloader:
            x1, y1 = x1.to(device), y1.to(device)

            y_pred1 = model(x1)
            lossL = calc_loss(y_pred1, y1)     # Dice_loss Used

            PA, Precision, Recall, F1, iou, dice = get_metrics(y_pred1, y1)
            PAs.append(PA.cpu().item())
            Precisions.append(Precision.cpu().item())
            Recalls.append(Recall.cpu().item())
            F1s.append(F1.cpu().item())
            ious.append(iou.cpu().item())
            dices.append(dice.cpu().item())  # 计算指标

            valid_loss += lossL.item() * x1.size(0)
            x_size1 = lossL.item() * x1.size(0)
        

    #######################################################
    # To write in Tensorboard
    #######################################################

    train_loss = train_loss / len(train_dataset)
    valid_loss = valid_loss / len(val_dataset)
    PAs = np.array(PAs).mean().item()
    Precisions = np.array(Precisions).mean().item()
    Recalls = np.array(Recalls).mean().item()
    F1s = np.array(F1s).mean().item()
    ious = np.array(ious).mean().item()
    dices = np.array(dices).mean().item()

    print(f'model: {model_name}')
    print('Training Loss: {:.6f} \tValidation Loss: {:.6f}'.format(train_loss, valid_loss))
    print('PA: {:.6f} \tPrecision: {:.6f} \tRecall: {:.6f} \tF1: {:.6f} \tIoU: {:.6f} \tDice: {:.6f}'.format(PAs, Precisions, Recalls, F1s, ious, dices))

    # 检查文件是否存在以及是否包含相同的模型信息
    if os.path.exists("model_info.json") and os.path.getsize("model_info.json") > 0:
        try:
            with open("model_info.json", "r") as f:
                existing_info = json.load(f)
        except json.JSONDecodeError:
            existing_info = {}
    else:
        existing_info = {}

    # 更新现有信息或添加新信息
    if model_name in existing_info:
        if "evaluation_metrics" in existing_info[model_name]:
            model_info = existing_info[model_name]["evaluation_metrics"]
        else:
            model_info = {}
    else:
        existing_info[model_name] = {}
        model_info = {}

    # 更新评估指标
    model_info.update({
        "PA": round(float(PAs), 6),
        "Precision": round(float(Precisions), 6),
        "Recall": round(float(Recalls), 6),
        "F1": round(float(F1s), 6),
        "IoU": round(float(ious), 6),
        "Dice": round(float(dices), 6),
        "train_loss": round(float(train_loss), 6),
        "valid_loss": round(float(valid_loss), 6)
    })

    existing_info[model_name]["evaluation_metrics"] = model_info

    # 将字典写入文件
    with open("model_info.json", "w") as f:
        json.dump(existing_info, f, indent=4)

if __name__=='__main__':
    model = U_NeXt_v4(in_channels=1, out_channels=1)
    batch_size = 8
    num_workers = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #######################################################
    # Dataset and Dataloader
    #######################################################

    train_dataset = WSSegmentation(r"/root/autodl-tmp/Dataset/WeldSeam",
                                    train=True,    
                                    txt_name="train.txt")
    val_dataset = WSSegmentation(r"/root/autodl-tmp/Dataset/WeldSeam",
                                    train=False,   
                                    txt_name="val.txt")

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    shuffle=True,
                                    pin_memory=True,
                                    collate_fn=train_dataset.collate_fn)
    valid_loader = torch.utils.data.DataLoader(val_dataset,
                                    batch_size=1,
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    collate_fn=val_dataset.collate_fn)
    
    evaluate_model(model, valid_loader, device)


