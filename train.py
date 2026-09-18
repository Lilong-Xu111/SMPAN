import torch
import os
import argparse
import json
import time
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch.optim as optim
from torch.amp import GradScaler, autocast
from torchvision import transforms
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from my_dataset import MyDataSet
from SMPAN import SMPAN
from utill import read_split_data, evaluate, calculate_metrics

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def validate_args(args):
    """验证命令行参数的合理性"""
    if args.img_h <= 0 or args.img_w <= 0:
        raise ValueError("图像高度和宽度必须为正数")
    if args.batch_size <= 0:
        raise ValueError("批量大小必须为正数")
    if args.epochs <= 0:
        raise ValueError("训练轮数必须为正数")
    if args.lr <= 0:
        raise ValueError("学习率必须为正数")
    if args.min_lr < 0:
        raise ValueError("最小学习率不能为负数")
    if args.min_lr > args.lr:
        raise ValueError("最小学习率不能大于初始学习率")
    if args.warmup_epochs < 0:
        raise ValueError("预热轮数不能为负数")
    if args.early_stop < 0:
        raise ValueError("早停耐心值不能为负数")
    if args.randaug_n < 1:
        raise ValueError("RandAugment操作数量必须至少为1")
    if args.randaug_m < 1:
        raise ValueError("RandAugment操作强度必须至少为1")
    if args.mixup_alpha <= 0:
        raise ValueError("Mixup的alpha参数必须为正数")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"数据集路径不存在: {args.data_path}")
    if args.weights and not os.path.exists(args.weights):
        raise FileNotFoundError(f"权重文件不存在: {args.weights}")


def create_model(num_classes: int = 30, img_size: tuple = (224, 224)):
    """创建模型并确保所有参数都在同一设备上"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SMPAN(num_classes=num_classes).to(device)

    # 验证所有参数都在GPU上
    for name, param in model.named_parameters():
        if param.device != device:
            logger.warning(f"参数 {name} 在 {param.device}，应移至 {device}")

    return model


def mixup_data(x, y, alpha=1.0, device=None):
    """
    实现Mixup数据增强
    :param x: 输入图像张量
    :param y: 输入标签张量
    :param alpha: Beta分布的参数
    :param device: 计算设备
    :return: 混合后的图像、混合后的标签、lambda值
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size()[0]
    if device is None:
        device = x.device

    # 生成随机索引
    index = torch.randperm(batch_size).to(device)

    # 混合图像和标签
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Mixup对应的损失计算
    :param criterion: 损失函数
    :param pred: 模型预测结果
    :param y_a: 原始标签
    :param y_b: 随机打乱的标签
    :param lam: 混合系数
    :return: 计算得到的损失
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def main(args):
    try:
        validate_args(args)
    except Exception as e:
        logger.error(f"参数验证失败: {e}")
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 创建保存目录
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./weights", exist_ok=True)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []  # 记录学习率
    }

    img_size = (args.img_h, args.img_w)

    data_transform = {
        "train": transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandAugment(num_ops=args.randaug_n, magnitude=args.randaug_m),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    logger.info(f"读取数据集: {args.data_path}")
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    # 使用pin_memory和persistent_workers优化数据加载
    train_dataset = MyDataSet(
        images_path=train_images_path,
        images_class=train_images_label,
        transform=data_transform["train"]
    )
    val_dataset = MyDataSet(
        images_path=val_images_path,
        images_class=val_images_label,
        transform=data_transform["val"]
    )

    batch_size = args.batch_size
    # 计算合适的数据加载器工作线程数
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    logger.info(f'使用 {nw} 个数据加载器工作线程')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=nw,
        persistent_workers=True if nw > 0 else False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        persistent_workers=True if nw > 0 else False
    )

    # 创建模型并确保所有参数都在GPU上
    model = create_model(
        num_classes=args.num_classes,
        img_size=(args.img_h, args.img_w)
    ).to(device)

    # 加载自定义权重
    if args.weights:
        logger.info(f"加载预训练权重: {args.weights}")
        checkpoint = torch.load(args.weights, map_location=device)

        # 处理可能的多GPU保存的权重
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        # 移除可能的module前缀
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        logger.info(f"成功加载预训练权重")

    # 冻结骨干层
    if args.freeze_layers:
        frozen_params = 0
        for name, param in model.named_parameters():
            if "backbone" in name:  # 根据实际结构调整
                param.requires_grad_(False)
                frozen_params += 1
        logger.info(f"冻结了 {frozen_params} 个骨干层参数")

    # 优化器和学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # 学习率预热 + 余弦退火
    warmup_epochs = args.warmup_epochs
    total_epochs = args.epochs

    # 预热阶段（线性增加）
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,  # 从0.1*lr开始
        total_iters=warmup_epochs
    )

    # 余弦退火阶段（学习率逐渐降低）
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=args.min_lr  # 最小学习率
    )

    # 混合精度训练设置
    use_amp = device.type == 'cuda'  # 仅在GPU上使用混合精度
    scaler = GradScaler(enabled=use_amp)

    best_val_acc = 0.0
    best_val_loss = float('inf')
    no_improvement_acc = 0  # 记录验证集准确率未提升的轮数
    no_improvement_loss = 0  # 记录验证集损失未降低的轮数
    early_stop_patience = args.early_stop  # 早停耐心值

    # 生成统一的时间戳（确保权重和Excel使用相同时间戳）
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    network_name = "net6_wo_ss_guidance"
    # 定义统一的基础文件名（与Excel保持一致）
    base_filename = f"{network_name}_{timestamp}_{args.epochs}epochs"

    logger.info(f"开始训练，总轮数: {total_epochs}")
    logger.info(f"使用数据增强: RandAugment(N={args.randaug_n}, M={args.randaug_m}) 和 Mixup(alpha={args.mixup_alpha})")

    # 定义损失函数
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        # 训练一个epoch - 手动实现训练循环以支持mixup
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for step, data in enumerate(train_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            # 应用Mixup增强
            images, labels_a, labels_b, lam = mixup_data(
                images, labels, args.mixup_alpha, device
            )

            optimizer.zero_grad()

            # 修复autocast参数错误，添加device_type
            with autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                outputs = model(images)
                # 使用mixup对应的损失计算
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

            # 混合精度训练
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 统计训练数据
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            # 计算Mixup的准确率
            correct += (lam * predicted.eq(labels_a.data).cpu().sum().float() +
                        (1 - lam) * predicted.eq(labels_b.data).cpu().sum().float())

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # 验证一个epoch
        val_loss, val_acc = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            epoch=epoch
        )

        # 学习率调度
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        history["lr"].append(current_lr)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc.item())
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info(f"Epoch [{epoch + 1}/{args.epochs}]  "
                    f"Train Loss: {train_loss:.4f}  "
                    f"Train Acc: {train_acc:.4f}  "
                    f"Val Loss: {val_loss:.4f}  "
                    f"Val Acc: {val_acc:.4f}  "
                    f"LR: {current_lr:.8f}")

        # 保存最佳模型（基于准确率）- 使用统一命名格式
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improvement_acc = 0
            acc_weights_path = os.path.join("./weights", f"{base_filename}_best_acc.pth")
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": cosine_scheduler.state_dict() if epoch >= warmup_epochs else warmup_scheduler.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss
            }, acc_weights_path)
            logger.info(f"保存准确率最佳模型: {acc_weights_path}, acc {val_acc:.6f}")
        else:
            no_improvement_acc += 1
            logger.info(f"准确率已有 {no_improvement_acc} 轮未提升")

        # 保存最佳模型（基于损失）- 使用统一命名格式
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_loss = 0
            loss_weights_path = os.path.join("./weights", f"{base_filename}_best_loss.pth")
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": cosine_scheduler.state_dict() if epoch >= warmup_epochs else warmup_scheduler.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss
            }, loss_weights_path)
            logger.info(f"保存损失最佳模型: {loss_weights_path}, loss {val_loss:.4f}")
        else:
            no_improvement_loss += 1
            logger.info(f"损失已有 {no_improvement_loss} 轮未降低")

        # 早停机制（同时考虑准确率和损失）
        if no_improvement_acc >= early_stop_patience and no_improvement_loss >= early_stop_patience:
            logger.info(f"早停触发: 准确率已有 {no_improvement_acc} 轮未提升，损失已有 {no_improvement_loss} 轮未降低")
            logger.info(f"早停于 epoch {epoch + 1}")
            break

    # 保存训练历史
    with open(f"./results/training_history_{timestamp}.json", "w") as f:
        json.dump(history, f, indent=4)

    # 绘制训练曲线
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss", color="blue")
    plt.plot(history["epoch"], history["val_loss"], label="Test Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Test Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history["epoch"], history["train_acc"], label="Train Acc", color="blue")
    plt.plot(history["epoch"], history["val_acc"], label="Test Acc", color="red")
    plt.axhline(y=best_val_acc, color='g', linestyle='--', label=f'Best Val Acc: {best_val_acc:.4f}')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Test Accuracy")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history["epoch"], history["lr"], label="Learning Rate", color="purple")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.title("Learning Rate Schedule")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"./results/training_curves_{timestamp}.png")
    plt.close()

    # 保存Excel表格（使用统一命名）
    df = pd.DataFrame(history)
    df["best_val_acc"] = best_val_acc
    df["best_val_loss"] = best_val_loss
    excel_path = os.path.join("./results", f"{base_filename}.xlsx")
    df.to_excel(excel_path, index=False, engine="openpyxl")
    logger.info(f"训练日志已保存至: {excel_path}")

    # 加载最佳模型并进行完整评估
    logger.info("加载最佳准确率模型进行最终评估")
    best_model = create_model(num_classes=args.num_classes, img_size=img_size)
    best_model.load_state_dict(torch.load(acc_weights_path)["model"])  # 使用保存的路径变量
    best_model.to(device)
    best_model.eval()

    # 假设你有类别名称列表
    class_names = [f"Class_{i}" for i in range(args.num_classes)]  # 替换为实际类别名称

    # 计算详细指标
    metrics = calculate_metrics(best_model, val_loader, device, args.num_classes, class_names)
    metrics_path = f"./results/metrics_{timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # 打印分类报告
    report = classification_report(
        metrics['true_labels'],
        metrics['pred_labels'],
        target_names=class_names,
        digits=4
    )
    logger.info(f"分类报告:\n{report}")

    logger.info(f"详细评估指标已保存至: {metrics_path}")
    logger.info(f"\n训练完成. 最佳验证准确率: {best_val_acc:.4f}, 最佳验证损失: {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MSCNet model for classification with RandAugment and Mixup")
    parser.add_argument('--num_classes', type=int, default=30, help="Number of classification classes")
    parser.add_argument('--epochs', type=int, default=100, help="Training epochs")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size (adjust based on GPU memory)")
    parser.add_argument('--lr', type=float, default=5e-5, help="Initial learning rate")
    parser.add_argument('--min_lr', type=float, default=1e-7, help="Minimum learning rate")
    parser.add_argument('--wd', type=float, default=5e-3, help="Weight decay")
    parser.add_argument('--data-path', type=str, default=r"C:\AID30\AID30",
                        help="Path to dataset directory")
    parser.add_argument('--weights', type=str, default='',
                        help="Path to pretrained weights (optional)")
    parser.add_argument('--freeze-layers', type=bool, default=False,
                        help="Freeze SMPAN backbone")
    parser.add_argument('--device', default='cuda:0', help="Training device (cuda:0 or cpu)")
    parser.add_argument('--img-h', type=int, default=256, help="Input image height")
    parser.add_argument('--img-w', type=int, default=256, help="Input image width")
    parser.add_argument('--warmup_epochs', type=int, default=5, help="Warmup epochs")
    parser.add_argument('--early_stop', type=int, default=100, help="Early stopping patience")

    # RandAugment参数
    parser.add_argument('--randaug_n', type=int, default=2, help="Number of RandAugment operations")
    parser.add_argument('--randaug_m', type=int, default=10, help="Magnitude of RandAugment operations")

    # Mixup参数
    parser.add_argument('--mixup_alpha', type=float, default=1, help="Alpha parameter for Mixup (Beta distribution)")

    opt = parser.parse_args()
    main(opt)
