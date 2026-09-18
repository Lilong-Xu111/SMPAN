import os
import random
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
from tqdm import tqdm


def read_split_data(data_path: str, val_ratio: float = 0.8):
    """
    读取数据集并分割为训练集和验证集

    Args:
        data_path: 数据集路径
        val_ratio: 验证集比例

    Returns:
        训练集图像路径, 训练集标签, 验证集图像路径, 验证集标签
    """
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(data_path), f"数据集路径 {data_path} 不存在"

    # 获取所有类别名称
    class_names = [cla for cla in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, cla))]
    # 按类别名称排序
    class_names.sort()
    # 生成类别索引字典
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}

    train_images_path = []  # 训练集图像路径
    train_images_label = []  # 训练集图像标签
    val_images_path = []  # 验证集图像路径
    val_images_label = []  # 验证集图像标签

    every_class_num = []  # 存储每个类别的样本数

    supported = ['.jpg']

    # supported = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    # supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀

    # 遍历每个类别
    for cla in class_names:
        cla_path = os.path.join(data_path, cla)
        # 获取所有支持的图像文件
        images = [os.path.join(cla_path, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[1] in supported]
        # 获取该类别的样本数
        every_class_num.append(len(images))
        # 按比例分割数据集
        val_path = random.sample(images, k=int(len(images) * val_ratio))

        for img_path in images:
            if img_path in val_path:  # 如果是验证集
                val_images_path.append(img_path)
                val_images_label.append(class_to_idx[cla])
            else:  # 如果是训练集
                train_images_path.append(img_path)
                train_images_label.append(class_to_idx[cla])

    print(f"数据集共发现 {sum(every_class_num)} 张图像")
    print(f"训练集: {len(train_images_path)} 张")
    print(f"验证集: {len(val_images_path)} 张")

    return train_images_path, train_images_label, val_images_path, val_images_label


def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None, use_amp=False):
    """
    训练一个epoch

    Args:
        model: 模型
        optimizer: 优化器
        data_loader: 数据加载器
        device: 设备
        epoch: 当前轮次
        scaler: 梯度缩放器(用于混合精度训练)
        use_amp: 是否使用混合精度训练

    Returns:
        平均损失, 准确率
    """
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    # 统计正确预测的样本数和总样本数
    correct = 0
    total = 0

    # 使用tqdm显示进度条
    data_loader = tqdm(data_loader, desc=f'Epoch [{epoch + 1}]')

    for step, (images, labels) in enumerate(data_loader):
        # 数据移至指定设备
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 使用混合精度训练
        if use_amp and scaler is not None:
            with torch.amp.autocast(device_type=device.type):  # 修改此处
                pred = model(images)
                loss = loss_function(pred, labels)

            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            # 正常训练流程
            pred = model(images)
            loss = loss_function(pred, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # 统计平均损失
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)

        # 统计准确率
        _, predicted = torch.max(pred, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 更新进度条
        data_loader.set_postfix(loss=mean_loss.item(),
                                acc=correct / total,
                                lr=optimizer.param_groups[0]["lr"])

    return mean_loss.item(), correct / total


def evaluate(model, data_loader, device, epoch):
    """
    评估模型

    Args:
        model: 模型
        data_loader: 数据加载器
        device: 设备
        epoch: 当前轮次

    Returns:
        平均损失, 准确率
    """
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)

    # 统计正确预测的样本数和总样本数
    correct = 0
    total = 0

    # 使用tqdm显示进度条
    data_loader = tqdm(data_loader, desc=f'Validation [{epoch + 1}]')

    with torch.no_grad():
        for step, (images, labels) in enumerate(data_loader):
            # 数据移至指定设备
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            pred = model(images)
            loss = loss_function(pred, labels)

            # 统计平均损失
            mean_loss = (mean_loss * step + loss.detach()) / (step + 1)

            # 统计准确率
            _, predicted = torch.max(pred, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 更新进度条
            data_loader.set_postfix(loss=mean_loss.item(),
                                    acc=correct / total)

    return mean_loss.item(), correct / total


def calculate_metrics(model, data_loader, device, num_classes, class_names=None):
    """
    计算模型在给定数据集上的详细评估指标

    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 计算设备
        num_classes: 类别数量
        class_names: 类别名称列表(可选)

    Returns:
        包含各种评估指标的字典
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 计算总体指标
    accuracy = accuracy_score(all_labels, all_preds)
    macro_precision = precision_score(all_labels, all_preds, average='macro')
    macro_recall = recall_score(all_labels, all_preds, average='macro')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_precision = precision_score(all_labels, all_preds, average='weighted')
    weighted_recall = recall_score(all_labels, all_preds, average='weighted')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

    # 计算每个类别的指标
    per_class_precision = precision_score(all_labels, all_preds, average=None)
    per_class_recall = recall_score(all_labels, all_preds, average=None)
    per_class_f1 = f1_score(all_labels, all_preds, average=None)

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 创建评估报告
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names if class_names is not None else [f'Class {i}' for i in range(num_classes)],
        output_dict=True
    )

    metrics = {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'per_class_metrics': {
            'precision': per_class_precision.tolist(),
            'recall': per_class_recall.tolist(),
            'f1': per_class_f1.tolist()
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'true_labels': all_labels.tolist(),
        'pred_labels': all_preds.tolist()
    }

    return metrics