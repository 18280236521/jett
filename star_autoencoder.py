"""
星体数据自编码器 — 基于重构Loss的异常检测
正常星体 → 重构Loss小，异常数据 → 重构Loss大
输入: 23x23 像素的 FITS 星体 stamps
"""

import os
import numpy as np
from astropy.io import fits
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# ==================== 配置 ====================
CONFIG = {
    # 数据路径
    "normal_dir": r"D:\Pycharm_proj\PythonProject\data\normal",      # 正常星体 FITS 目录
    "anomaly_dir": r"None",    # 异常数据 FITS 目录（验证用，可选）
    "model_save_path": "./star_ae_model.pth",

    # 图像参数
    "image_size": 23,
    "in_channels": 1,

    # 训练参数
    "batch_size": 64,
    "epochs": 200,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "val_split": 0.1,
    "num_workers": 0,

    # 异常检测阈值（训练集重构Loss的百分位数）
    "threshold_percentile": 95,

    # 设备
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# ==================== 数据集 ====================
class StarStampDataset(Dataset):
    """从 FITS 文件目录加载 23x23 星体 stamps"""

    def __init__(self, data_dir):
        self.files = sorted(Path(data_dir).glob("*.fits"))
        if not self.files:
            raise FileNotFoundError(f"在 {data_dir} 中没有找到 .fits 文件")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        stamp = fits.getdata(self.files[idx]).astype(np.float32)
        # 归一化到 [0, 1]
        smin, smax = stamp.min(), stamp.max()
        if smax - smin > 1e-8:
            stamp = (stamp - smin) / (smax - smin)
        else:
            stamp = np.zeros_like(stamp)
        # 添加 channel 维度: (1, 23, 23)
        return torch.from_numpy(stamp).unsqueeze(0)


# ==================== 模型 ====================
class StarAutoencoder(nn.Module):
    """23x23 卷积自编码器 — 轻量级对称结构"""

    def __init__(self):
        super().__init__()

        # ---------- Encoder ----------
        self.encoder = nn.Sequential(
            # 23 -> 12
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # 12 -> 6
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # 6 -> 3
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 3 -> 2
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 3x3->2x2
            nn.ReLU(True),
        )
        self.latent_dim = 128 * 2 * 2  # 512

        # ---------- Decoder ----------
        self.decoder = nn.Sequential(
            # 2 -> 3
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 3 -> 6
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # 6 -> 12
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # 12 -> 23
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.Sigmoid(),  # 输出 [0,1]
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# ==================== 训练 ====================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon = model(batch)
        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        recon = model(batch)
        total_loss += criterion(recon, batch).item() * batch.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def compute_recon_errors(model, loader, device):
    """返回每个样本的逐像素MSE"""
    model.eval()
    errors = []
    criterion = nn.MSELoss(reduction='none')
    for batch in loader:
        batch = batch.to(device)
        recon = model(batch)
        # mean over (C, H, W) per sample
        mse = criterion(recon, batch).mean(dim=(1, 2, 3))
        errors.append(mse.cpu().numpy())
    return np.concatenate(errors)


# ==================== 异常检测 ====================
class AnomalyDetector:
    """基于重构误差的异常检测器"""

    def __init__(self, model, threshold, device):
        self.model = model
        self.threshold = threshold
        self.device = device

    @torch.no_grad()
    def predict(self, dataloader):
        """返回 (预测标签, 重构误差) — 1=异常, 0=正常"""
        self.model.eval()
        errors = compute_recon_errors(self.model, dataloader, self.device)
        labels = (errors > self.threshold).astype(int)
        return labels, errors


# ==================== 可视化 ====================
def plot_reconstructions(model, dataset, device, indices=None, save_path="recon_samples.png"):
    if indices is None:
        indices = [0, 1, 2, 3, 4, 5]
    model.eval()
    n = len(indices)
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i, idx in enumerate(indices):
        x = dataset[idx].unsqueeze(0).to(device)
        r = model(x).cpu().squeeze().numpy()
        x_np = x.cpu().squeeze().numpy()
        axes[0, i].imshow(x_np, cmap='gray')
        axes[0, i].set_title(f"Input {idx}")
        axes[0, i].axis('off')
        axes[1, i].imshow(r, cmap='gray')
        axes[1, i].set_title(f"Recon {idx}")
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"重构对比图已保存至 {save_path}")


def plot_loss_curve(train_losses, val_losses, save_path="loss_curve.png"):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss", linewidth=1)
    plt.plot(val_losses, label="Val Loss", linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"Loss 曲线已保存至 {save_path}")


def plot_error_distribution(normal_errors, anomaly_errors=None, threshold=None,
                            save_path="error_distribution.png"):
    plt.figure(figsize=(8, 4))
    plt.hist(normal_errors, bins=60, alpha=0.7, label="Normal", density=True, color='green')
    if anomaly_errors is not None:
        plt.hist(anomaly_errors, bins=60, alpha=0.7, label="Anomaly", density=True, color='red')
    if threshold is not None:
        plt.axvline(threshold, color='black', linestyle='--', linewidth=1.5,
                    label=f"Threshold={threshold:.6f}")
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"误差分布图已保存至 {save_path}")


# ==================== 主流程 ====================
def main():
    cfg = CONFIG
    device = torch.device(cfg["device"])
    print(f"使用设备: {device}")

    # ---- 1. 加载正常数据 ----
    print(f"\n加载正常星体数据: {cfg['normal_dir']}")
    dataset = StarStampDataset(cfg["normal_dir"])
    print(f"共 {len(dataset)} 个正常样本")

    # 划分训练/验证集
    n_val = int(len(dataset) * cfg["val_split"])
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"])
    full_loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg["num_workers"])

    # ---- 2. 构建模型 ----
    model = StarAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("开始训练...\n")

    # ---- 3. 训练 ----
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), cfg["model_save_path"])

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{cfg['epochs']} | Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | Best Val: {best_val_loss:.6f}")

    print(f"\n训练完成，最佳验证Loss: {best_val_loss:.6f}")
    model.load_state_dict(torch.load(cfg["model_save_path"], map_location=device))

    # ---- 4. 计算正常数据的重构误差 & 阈值 ----
    print("\n计算正常训练集重构误差...")
    normal_errors = compute_recon_errors(model, full_loader, device)
    threshold = np.percentile(normal_errors, cfg["threshold_percentile"])
    np.save(cfg["model_save_path"].replace(".pth", "_threshold.npy"), threshold)
    print(f"正常数据重构误差: mean={normal_errors.mean():.6f}, std={normal_errors.std():.6f}")
    print(f"异常阈值 (P{cfg['threshold_percentile']}): {threshold:.6f}")

    # ---- 5. 如有异常数据，评估检测性能 ----
    anomaly_errors = None
    if cfg["anomaly_dir"] and os.path.isdir(cfg["anomaly_dir"]):
        print(f"\n加载异常数据: {cfg['anomaly_dir']}")
        anomaly_ds = StarStampDataset(cfg["anomaly_dir"])
        anomaly_loader = DataLoader(anomaly_ds, batch_size=cfg["batch_size"], shuffle=False,
                                    num_workers=cfg["num_workers"])
        anomaly_errors = compute_recon_errors(model, anomaly_loader, device)
        print(f"异常数据重构误差: mean={anomaly_errors.mean():.6f}, std={anomaly_errors.std():.6f}")

        # 构建标签计算 AUC
        y_true = np.concatenate([np.zeros(len(normal_errors)), np.ones(len(anomaly_errors))])
        y_score = np.concatenate([normal_errors, anomaly_errors])
        auc = roc_auc_score(y_true, y_score)
        print(f"ROC-AUC: {auc:.4f}")

        # 阈值检测准确率
        tp = (anomaly_errors > threshold).sum()
        fn = (anomaly_errors <= threshold).sum()
        tn = (normal_errors <= threshold).sum()
        fp = (normal_errors > threshold).sum()
        acc = (tp + tn) / len(y_true)
        print(f"TP: {tp} | FN: {fn} | TN: {tn} | FP: {fp}")
        print(f"准确率: {acc:.4f} | 召回率(异常): {tp/(tp+fn):.4f}")

    # ---- 6. 可视化 ----
    print("\n生成可视化...")
    plot_loss_curve(train_losses, val_losses)
    plot_reconstructions(model, dataset, device)
    plot_error_distribution(normal_errors, anomaly_errors, threshold)

    # ---- 7. 输出使用说明 ----
    print("\n" + "=" * 55)
    print("异常检测器已就绪，使用示例:")
    print("-" * 55)
    print("from star_autoencoder import detect")
    print("results = detect('path/to/test_fits_dir',")
    print("                 model_path='star_ae_model.pth')")
    print("-" * 55)
    print("results: List[Dict]  每张图的预测结果")
    print("  - file:     文件名")
    print("  - label:    0=正常, 1=异常")
    print("  - mse:      重构误差")
    print("=" * 55)


# ==================== 推理接口 ====================
def detect(test_dir, model_path="star_ae_model.pth", device=None):
    """
    对指定目录下的 FITS 文件进行异常检测
    返回: [{"file": str, "label": 0/1, "mse": float}, ...]
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # 加载训练集的阈值
    threshold_file = model_path.replace(".pth", "_threshold.npy")
    if os.path.exists(threshold_file):
        threshold = float(np.load(threshold_file))
    else:
        raise FileNotFoundError(
            f"阈值文件 {threshold_file} 不存在，请先运行 main() 训练并保存阈值"
        )

    model = StarAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset = StarStampDataset(test_dir)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    errors = compute_recon_errors(model, loader, device)

    results = []
    for i, f in enumerate(dataset.files):
        results.append({
            "file": str(f),
            "label": int(errors[i] > threshold),
            "mse": float(errors[i]),
        })
    return results


if __name__ == "__main__":
    main()
