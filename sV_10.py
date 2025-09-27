import os
import pandas as pd
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from scipy.io import loadmat
import re
from torchvision import transforms
import time

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# from RES_CAT_model import Res_Cat
from RES_KongDong import Res_Cat




# 定义训练的设备
device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
print(device)
# 准备数据集
class MyData(Dataset):

    def __init__(self, root_dir, label_dir, transform=None):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)
        self.img_path = os.listdir(self.path)
        self.transform = transform

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        file_name = str(img_name)
        # 编译正则表达式，匹配文件名的模式
        pattern = re.compile(r'img_([\d.]+)v_([\d.]+)t_([\d.]+)s_([\d.]+)Z_([\d.]+)\.mat')
        # 尝试匹配文件名
        match = pattern.match(file_name)
        v = 0
        t = 0
        s = 0
        z = 0
        # 如果匹配成功A
        if match:
            # 提取速度和大小
            v = float(match.group(2))
            t = float(match.group(3))
            s = float(match.group(4))
            z = float(match.group(5))
        if z != 0:
            v = v/1.5
            t = t/0.5
            s = s/15
            z = np.log10(z)/11
        else:
            v = v / 1.5
            t = t / 0.5
            s = s / 15

        add = [v, t, s, z]
        add = torch.tensor(add)

        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        wenjian = loadmat(img_item_path)  # 读取mat文件
        img = wenjian['heatmaps']
        img = img.astype(np.float32)
        img_1 = wenjian['heatmaps5']
        img_1 = img_1.astype(np.float32)
        # img = np.transpose(img, axes=(1, 0, 2))
        label = self.label_dir
        if self.transform:
            img = transform(img)
            img_1 = transform_1(img_1)
        return img, img_1, label, add

    def __len__(self):
        return len(self.img_path)

def adjust_learning_rate(optimizer, epoch, start_lr):
    lr = start_lr * (0.6 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
# 准备数据集Mean: [0.29753032 0.31325793 0.33182207 0.34455618 0.3359477 ]td Dev: [0.09510516 0.11785412 0.16713324 0.15151253 0.13168932]
# Mean: [] Std Dev: []
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.29382405, 0.31110755, 0.333575, 0.3448717, 0.3354617), (0.08475687, 0.10914939, 0.16213238, 0.14575034, 0.12631622))])
transform_1 = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.2978428, 0.3141507, 0.33672965, 0.345798, 0.337347), (0.07503843, 0.09711462, 0.14508125, 0.1299235, 0.11258475))])

train_root_dir = "CGDS_10/train"
sun_label_dir = "1"
wang_label_dir = "2"
yu_label_dir = "3"
bi_label_dir = "4"
zhang_label_dir = "5"
zhao_label_dir = "6"
xuan_label_dir = "7"
sang_label_dir = "8"
shi_label_dir = "9"
liu_label_dir = "0"
'''
zhangs_label_dir = "10"
lik_label_dir = "11"
liux_label_dir = "12"
hu_label_dir = "13"
wangw_label_dir = "14"
liul_label_dir = "15"
xuj_label_dir = "16"
shen_label_dir = "17"
xiaoy_label_dir = "18"
gao_label_dir = "19"
'''
train_sun_data = MyData(train_root_dir, sun_label_dir, transform=transform)
train_wang_data = MyData(train_root_dir, wang_label_dir, transform=transform)
train_yu_data = MyData(train_root_dir, yu_label_dir, transform=transform)
train_bi_data = MyData(train_root_dir, bi_label_dir, transform=transform)
train_zhang_data = MyData(train_root_dir, zhang_label_dir, transform=transform)
train_zhao_data = MyData(train_root_dir, zhao_label_dir, transform=transform)
train_xuan_data = MyData(train_root_dir, xuan_label_dir, transform=transform)
train_sang_data = MyData(train_root_dir, sang_label_dir, transform=transform)
train_shi_data = MyData(train_root_dir, shi_label_dir, transform=transform)
train_liu_data = MyData(train_root_dir, liu_label_dir, transform=transform)
'''
train_zhangs_data = MyData(train_root_dir, zhangs_label_dir, transform=transform)
train_lik_data = MyData(train_root_dir, lik_label_dir, transform=transform)
train_liux_data = MyData(train_root_dir, liux_label_dir, transform=transform)
train_hu_data = MyData(train_root_dir, hu_label_dir, transform=transform)
train_wangw_data = MyData(train_root_dir, wangw_label_dir, transform=transform)
train_liul_data = MyData(train_root_dir, liul_label_dir, transform=transform)
train_xuj_data = MyData(train_root_dir, xuj_label_dir, transform=transform)
train_shen_data = MyData(train_root_dir, shen_label_dir, transform=transform)
train_xiaoy_data = MyData(train_root_dir, xiaoy_label_dir, transform=transform)
train_gao_data = MyData(train_root_dir, gao_label_dir, transform=transform)
'''
train_data = train_sun_data + train_wang_data + train_yu_data + train_bi_data + train_zhang_data + train_zhao_data + train_xuan_data + train_sang_data + train_shi_data + train_liu_data # + train_zhangs_data + train_lik_data + train_liux_data + train_hu_data + train_wangw_data + train_liul_data + train_xuj_data + train_shen_data + train_xiaoy_data + train_gao_data
# train_data = train_sun_data + train_wang_data + train_yu_data + train_zhang_data

val_root_dir = "CGDS_10/val"
val_sun_data = MyData(val_root_dir, sun_label_dir, transform=transform)
val_wang_data = MyData(val_root_dir, wang_label_dir, transform=transform)
val_yu_data = MyData(val_root_dir, yu_label_dir, transform=transform)
val_bi_data = MyData(val_root_dir, bi_label_dir, transform=transform)
val_zhang_data = MyData(val_root_dir, zhang_label_dir, transform=transform)
val_zhao_data = MyData(val_root_dir, zhao_label_dir, transform=transform)
val_xuan_data = MyData(val_root_dir, xuan_label_dir, transform=transform)
val_sang_data = MyData(val_root_dir, sang_label_dir, transform=transform)
val_shi_data = MyData(val_root_dir, shi_label_dir, transform=transform)
val_liu_data = MyData(val_root_dir, liu_label_dir, transform=transform)
'''
val_zhangs_data = MyData(val_root_dir, zhangs_label_dir, transform=transform)
val_lik_data = MyData(val_root_dir, lik_label_dir, transform=transform)
val_liux_data = MyData(val_root_dir, liux_label_dir, transform=transform)
val_hu_data = MyData(val_root_dir, hu_label_dir, transform=transform)
val_wangw_data = MyData(val_root_dir, wangw_label_dir, transform=transform)
val_liul_data = MyData(val_root_dir, liul_label_dir, transform=transform)
val_xuj_data = MyData(val_root_dir, xuj_label_dir, transform=transform)
val_shen_data = MyData(val_root_dir, shen_label_dir, transform=transform)
val_xiaoy_data = MyData(val_root_dir, xiaoy_label_dir, transform=transform)
val_gao_data = MyData(val_root_dir, gao_label_dir, transform=transform)
'''

test_data = val_sun_data + val_wang_data + val_yu_data + val_bi_data + val_zhang_data + val_zhao_data + val_xuan_data + val_sang_data + val_shi_data + val_liu_data # + val_zhangs_data + val_lik_data + val_liux_data + val_hu_data + val_wangw_data + val_liul_data + val_xuj_data + val_shen_data + val_xiaoy_data + val_gao_data
# test_data = val_sun_data + val_wang_data + val_yu_data + val_zhang_data

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 如果data_size = 10,训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用DataLoader加载数据集
train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)



VMD = Res_Cat()
VMD = VMD.to(device)
# 创建注意力机制

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
start_lr = 1e-2
optimizer = torch.optim.SGD(VMD.parameters(), lr=start_lr, weight_decay=0.01)

# Track last five models, the best model based on accuracy, and the best model based on test loss
best_accuracy = 0
best_accuracy_model = None
best_test_loss = float('inf')
best_test_loss_model = None

# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 1
# 训练次数
epoch = 100


a = 0
a1 = 0

# Data structures to track metrics
epoch_loss = []
epoch_accuracy = []

# Tensorboard writer
writer = SummaryWriter("./logs_train_GaitCube")

# Training loop (your code with modifications for tracking and saving metrics)
for i in range(epoch):
    print("-------第{}轮训练开始-------".format(i + 1))
    adjust_learning_rate(optimizer, i, start_lr)
    print("Epoch:{}  Lr:{:.2E}".format(i, optimizer.state_dict()['param_groups'][0]['lr']))

    # Training step
    VMD.train()
    total_train_loss = 0
    for data in train_dataloader:
        start_time = time.time()
        imgs, img_1s, labels, adds = data
        adds = adds.to(imgs.dtype)
        adds = adds.to(device)
        imgs = imgs.to(device)
        img_1s = img_1s.to(device)
        int_list = []
        for item in labels:
            int_list.append(int(item))
        labels = torch.tensor(int_list, dtype=torch.long).to(device)

        outputs = VMD(imgs, adds, img_1s)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        total_train_step = total_train_step + 1
        if total_train_step % 1000 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), global_step=total_train_step)

    # Evaluation step
    VMD.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            # start_time = time.time()
            imgs, img_1s, labels, adds = data
            adds = adds.to(imgs.dtype)
            adds = adds.to(device)
            imgs = imgs.to(device)
            img_1s = img_1s.to(device)
            int_list = []
            for item in labels:
                int_list.append(int(item))
            labels = torch.tensor(int_list, dtype=torch.long).to(device)
            outputs = VMD(imgs, adds, img_1s)
            loss = loss_fn(outputs, labels)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == labels).sum()
            total_accuracy += accuracy
            # end_time = time.time()
            # print(end_time - start_time)
    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy / test_data_size))

    total_test_step = total_test_step + 1
    b = (total_accuracy / test_data_size).clone().detach()
    # 或者更推荐直接转换为Python标量
    b = (total_accuracy / test_data_size)  # 直接获取数值
    b = b.unsqueeze(0)
    if i == 0:
        a = b
    else:
        a = torch.cat((a, b), dim=0)

    b1 = torch.tensor(total_test_loss)
    b1 = b1.unsqueeze(0)
    if i == 0:
        a1 = b1
    else:
        a1 = torch.cat((a1, b1), dim=0)
    # 保存每一轮的结果
    # Log and save metrics
    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_test_loss = total_test_loss / len(test_dataloader)
    avg_accuracy = total_accuracy / len(test_data)

    epoch_loss.append((avg_train_loss, avg_test_loss))
    epoch_accuracy.append(avg_accuracy.cpu().item())

    print(f"Epoch {i + 1} - Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    writer.add_scalar("train_loss", avg_train_loss, i)
    writer.add_scalar("test_loss", avg_test_loss, i)
    writer.add_scalar("test_accuracy", avg_accuracy, i)


    # Save the last five models
    if i >= 30:
        if epoch - i <= 2:
            model_checkpoint_path = f"./Model10/l4_model_epoch_{i + 1}.pth"
            model_checkpoint_path_all = f"./Model10/l4_model_epoch_{i + 1}_all.pth"
            torch.save(VMD, model_checkpoint_path_all)
            torch.save(VMD.state_dict(), model_checkpoint_path)

    # Save the best model based on accuracy
    if avg_accuracy > best_accuracy:
        best_accuracy = avg_accuracy
        best_accuracy_model_path_all = f"./Model10/l4_best_accuracy_model_all.pth"
        best_accuracy_model_path = f"./Model10/l4_best_accuracy_model.pth"
        torch.save(VMD, best_accuracy_model_path_all)
        torch.save(VMD.state_dict(), best_accuracy_model_path)




    # Save the best model based on test loss
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        best_test_loss_model_path_all = f"./Model10/l4_best_test_loss_model_all.pth"
        best_test_loss_model_path = f"./Model10/l4_best_test_loss_model.pth"
        torch.save(VMD, best_test_loss_model_path_all)
        torch.save(VMD.state_dict(), best_test_loss_model_path)

# Save metrics to CSV
metrics_df = pd.DataFrame({
    'Epoch': range(1, epoch + 1),
    'Train Loss': [x[0] for x in epoch_loss],
    'Test Loss': [x[1] for x in epoch_loss],
    'Test Accuracy': epoch_accuracy
})

# Save to CSV file

metrics_df.to_csv('./Model10/l4training_metrics.csv', index=False)
# Print out the loss and accuracy for each epoch
print("------- Training Finished! -------")
print("Epoch-wise Training and Test Loss, Accuracy:")
for i in range(epoch):
    print(f"Epoch {i + 1} - Train Loss: {epoch_loss[i][0]:.4f}, Test Loss: {epoch_loss[i][1]:.4f}, Test Accuracy: {epoch_accuracy[i]:.4f}")

# Close the writer
writer.close()

print("每一轮的准确率：", a)
print("每一轮的Loss：", a1)







