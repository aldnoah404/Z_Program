import torch
import random
import numpy as np
from model import U_NeXt_v1
from dataset import WSSegmentation
import os
import shutil
from torch.utils.tensorboard import SummaryWriter
from loss import calc_loss, threshold_predictions_p, get_metrics
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

#######################################################
# Setting device to GPU if available
#######################################################
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#######################################################
# Setting the basic paramters of the model
#######################################################
batch_size = 8
print('batch_size = ' + str(batch_size))

epoch = 200
print('epoch = ' + str(epoch))

random_seed = random.randint(1, 100)
print('random_seed = ' + str(random_seed))

num_workers = 2
print('num_workers = ' + str(num_workers))

valid_loss_min = np.inf
num_workers = 0
lossT = []    # 训练损失
lossL = []    # 验证损失
lossL.append(np.inf)
lossT.append(np.inf)
n_iter = 1
i_valid = 0

#######################################################
# Setting up the model
#######################################################

model_test = U_NeXt_v1(in_channels=1, out_channels=1)
model_test.to(device)
model_name = model_test.__class__.__name__ # 获取模型名称


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


#######################################################
# Using Adam as Optimizer
#######################################################

initial_lr = 1e-4
opt = torch.optim.Adam(model_test.parameters(), lr=initial_lr)

MAX_STEP = 64
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_STEP, eta_min=1e-5)

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


#######################################################
# Creating a Folder for every data of the program
#######################################################

New_folder = './exp/' + model_name

if os.path.exists(New_folder) and os.path.isdir(New_folder):
    shutil.rmtree(New_folder)
try:
    os.makedirs(New_folder)
except OSError:
    print("Creation of the main directory '%s' failed " % New_folder)
else:
    print("Successfully created the main directory '%s' " % New_folder)
    

#######################################################
# Training loop
#######################################################

writer1 = SummaryWriter(log_dir='./exp/log')
iter_id = 1
for i in range(epoch):

    train_loss = 0.0
    valid_loss = 0.0


    #######################################################
    # Training Data
    #######################################################

    model_test.train()

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        writer1.add_scalar('lr', scheduler.get_last_lr()[0], iter_id)

        opt.zero_grad()
        y_pred = model_test(x)
        lossT = calc_loss(y_pred, y)     # Dice_loss Used   这里的lossT准确来说是bce+dice
        # lossT += group_lasso(model_test, lambda_lasso=1e-4)

        train_loss += lossT.item() * x.size(0)
        lossT.backward()

        opt.step()
        x_size = lossT.item() * x.size(0)

        writer1.add_scalar('train_loss_iter', lossT, iter_id)
        iter_id += 1

    scheduler.step()  # 更新学习率

    # 记录模型参数和梯度的变化，以便在 TensorBoard 中进行可视化分析。
    #    for name, param in model_test.named_parameters():
    #        name = name.replace('.', '/')
    #        writer1.add_histogram(name, param.data.cpu().numpy(), i + 1)
    #        writer1.add_histogram(name + '/grad', param.grad.data.cpu().numpy(), i + 1)


    #######################################################
    # Validation Step
    #######################################################

    model_test.eval()
    PAs = []
    Precisions = []
    Recalls = []
    F1s = []
    ious = []
    dices = []
    with torch.no_grad():  
        for x1, y1 in valid_loader:
            x1, y1 = x1.to(device), y1.to(device)

            y_pred1 = model_test(x1)
            lossL = calc_loss(y_pred1, y1)     # Dice_loss Used

            PA, Precision, Recall, F1, iou, dice = get_metrics(y_pred1, y1)
            PAs.append(PA.cpu())
            Precisions.append(Precision.cpu())
            Recalls.append(Recall.cpu())
            F1s.append(F1.cpu())
            ious.append(iou.cpu())
            dices.append(dice.cpu())  # 计算指标

            valid_loss += lossL.item() * x1.size(0)
            x_size1 = lossL.item() * x1.size(0)
        

    #######################################################
    # To write in Tensorboard
    #######################################################

    train_loss = train_loss / len(train_dataset)
    valid_loss = valid_loss / len(val_dataset)
    PAs = np.array(PAs).mean()
    Precisions = np.array(Precisions).mean()
    Recalls = np.array(Recalls).mean()
    F1s = np.array(F1s).mean()
    ious = np.array(ious).mean()
    dices = np.array(dices).mean()

    if (i+1) % 1 == 0:
        print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(i + 1, epoch, train_loss,
                                                                                      valid_loss))
        print('PA: {:.6f} \tPrecision: {:.6f} \tRecall: {:.6f} \tF1: {:.6f} \tIoU: {:.6f} \tDice: {:.6f}'.format(PAs, Precisions, Recalls, F1s, ious, dices))
        writer1.add_scalar('Loss/train loss', train_loss, n_iter)
        writer1.add_scalar('Loss/val loss', valid_loss, n_iter)
        writer1.add_scalars(main_tag='Loss/train_val loss',
                                                tag_scalar_dict={'train': train_loss,
                                                                                    'val': valid_loss},
                                                global_step=n_iter)
        writer1.add_scalar('PA', PAs, n_iter)
        writer1.add_scalar('Precision', Precisions, n_iter)
        writer1.add_scalar('Recall', Recalls, n_iter)
        writer1.add_scalar('F1', F1s, n_iter)
        writer1.add_scalar('IoU', ious, n_iter)
        writer1.add_scalar('Dice', dices, n_iter)
        n_iter += 1


    #######################################################
    # save model
    #######################################################

    if valid_loss <= valid_loss_min:

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model '.format(valid_loss_min, valid_loss))
        torch.save(model_test.state_dict(),f'./exp/{model_name}/' +
                                              'best_sslse_epoch_' + str(epoch)
                                              + '_batchsize_' + str(batch_size) + '.pth')
        valid_loss_min = valid_loss
    
    # 每隔20个epoch保存一次模型
    if (i + 1) % 20 == 0:
        torch.save(model_test.state_dict(), f'./exp/{model_name}/' +
                   'checkpoint_epoch_' + str(i + 1) +
                   '_batchsize_' + str(batch_size) + '.pth')


#######################################################
# closing the tensorboard writer
#######################################################

writer1.close()


#######################################################
# Loading the model
#######################################################

if torch.cuda.is_available():
    torch.cuda.empty_cache()

model_test.load_state_dict(torch.load(f'./exp/{model_name}/' +
                   'best_sslse_epoch_' + str(epoch)
                   + '_batchsize_' + str(batch_size) + '.pth'))

model_test.eval()


#######################################################
# Visualize the segmentation effect of the model 
#   on the partial sample of the verification
#######################################################

indexs = random.sample(range(len(val_dataset)), 10)
for index in indexs:
    img, target = val_dataset[index]
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model_test(img)
    pred = F.sigmoid(pred)
    pred = pred[0][0].detach().cpu().numpy()
    pred = threshold_predictions_p(pred, 0.3)
    pred = pred.astype(np.uint8)
    pred = Image.fromarray(pred)
    plt.imshow(pred)
    plt.show()