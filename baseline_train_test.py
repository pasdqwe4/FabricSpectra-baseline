import os
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn import metrics
from torch.optim.lr_scheduler import MultiStepLR
import torch_optimizer as optim

from data_utils.data_process import NIRDataset_train, NIRDataset_test
from model.baselines import *
from model.swin_v2 import Swinv2_t
from model.tsai import *
from model_utils import setup_seed
from chronos import ChronosPipeline

# -------------------- Setup --------------------
setup_seed(0)

# Select model
# model = gmlp()
# model = tss()
# model = Swin_t()
# model = convnextv2_tiny()
# model = ChronosPipeline.from_pretrained(
#     "amazon/chronos-t5-tiny",
#     device_map="cuda",
#     torch_dtype=torch.bfloat16,
# )
model = Swinv2_t()

# model = nn.DataParallel(model)  # Uncomment if using multiple GPUs
model = model.cuda()

# Optimizer and scheduler
optimizer = optim.Lamb(model.parameters(), lr=0.001, weight_decay=0.001)
scheduler = MultiStepLR(optimizer, milestones=[40, 70, 90], gamma=0.1)

# -------------------- Utilities --------------------
def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_checkpoint(state, path, filename='checkpoint.pth.tar'):
    save_path = os.path.join(path, filename)
    torch.save(state, save_path)
    shutil.copyfile(save_path, os.path.join(path, 'model_best.pth.tar'))

def save_checkpoint_mse(state, path, filename='checkpoint_mae.pth.tar'):
    save_path = os.path.join(path, filename)
    torch.save(state, save_path)
    shutil.copyfile(save_path, os.path.join(path, 'model_best_mse.pth.tar'))

def save_checkpoint_time(state, path, filename='checkpoint_time.pth.tar'):
    save_path = os.path.join(path, filename)
    torch.save(state, save_path)

def cal_error_acc(output, label, threshold):
    within_threshold = (np.abs(output - label) <= threshold).all(axis=1)
    return np.mean(within_threshold)

def threshold_predictions(logits, threshold=0.5):
    return (torch.sigmoid(logits) > threshold).int()

# -------------------- Training --------------------
def train(train_loader, epoch):
    model.train()
    t = time.time()

    for batch_idx, (x, y, comp) in enumerate(train_loader):
        x, y, comp = x.cuda(), y.cuda(), comp.cuda()

        optimizer.zero_grad()
        mlc_out, reg_out = model(x)

        loss_mlc = F.multilabel_soft_margin_loss(mlc_out, y)
        loss_reg = F.l1_loss(reg_out, comp)
        total_loss = loss_mlc + loss_reg

        total_loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            final_pred = threshold_predictions(mlc_out).cpu().numpy()
            acc = metrics.accuracy_score(y.cpu().numpy(), final_pred)
            mse = metrics.mean_squared_error(comp.cpu().numpy(), reg_out.cpu().detach().numpy())

            print("####################################################################################")
            print(f"Epoch: {epoch+1:04d}, Iteration: {batch_idx+1:04d}")
            print(f"Loss MLC: {loss_mlc.item():.6f}, Loss Reg: {loss_reg.item():.6f}")
            print(f"Total Loss: {total_loss.item():.6f}, Accuracy: {acc:.4f}, MSE: {mse:.4f}")
            print(f"Time: {time.time() - t:.4f}s")
            print("####################################################################################")

        t = time.time()

# -------------------- Validation --------------------
def validate(val_loader, epoch):
    model.eval()
    t = time.time()

    total_acc, total_hamming, total_mse, count = 0, 0, 0, 0
    all_preds, all_labels = [], []

    for batch_idx, (x, y, comp) in enumerate(val_loader):
        x, y, comp = x.cuda(), y.cuda(), comp.cuda()

        with torch.no_grad():
            mlc_out, reg_out = model(x)

        preds = threshold_predictions(mlc_out).cpu().numpy()
        reg_preds = reg_out.cpu().numpy()
        y_np = y.cpu().numpy()
        comp_np = comp.cpu().numpy()

        n = x.size(0)
        acc = metrics.accuracy_score(y_np, preds)
        mse = metrics.mean_squared_error(comp_np, reg_preds)
        hamming = metrics.hamming_loss(y_np, preds)

        total_acc += acc * n
        total_mse += mse * n
        total_hamming += hamming * n
        count += n

        all_preds.append(reg_preds)
        all_labels.append(comp_np)

        if batch_idx % 10 == 0:
            print(f"Epoch: {epoch+1:04d}, Iteration: {batch_idx+1:04d}")
            print(f"MSE: {mse:.4f}, Accuracy: {acc:.4f}, Hamming Loss: {hamming:.4f}, Time: {time.time() - t:.4f}s")

        t = time.time()

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    return (
        total_acc / count,
        total_mse / count,
        total_hamming / count,
        *[cal_error_acc(all_preds, all_labels, th) for th in [0, 1, 2, 3, 5, 10]],
    )

# -------------------- Main --------------------
def main():
    save_path = "save_path"
    check_dir(save_path)

    root = "root_path"
    train_set = NIRDataset_train(
        os.path.join(root, "train_x.npy"),
        os.path.join(root, "train_y.npy"),
        os.path.join(root, "train_c.npy"),
    )
    val_set = NIRDataset_test(
        os.path.join(root, "test_x.npy"),
        os.path.join(root, "test_y.npy"),
        os.path.join(root, "test_c.npy"),
    )

    train_loader = Data.DataLoader(train_set, batch_size=1024, shuffle=True, num_workers=1)
    val_loader = Data.DataLoader(val_set, batch_size=1024, shuffle=False, num_workers=1)

    best = {
        "acc": 0, "mse": float('inf'), "hamming": float('inf'),
        "a0": 0, "a1": 0, "a2": 0, "a3": 0, "a5": 0, "a10": 0
    }

    for epoch in range(100):
        train(train_loader, epoch)
        scheduler.step()

        acc, mse, hamming, a0, a1, a2, a3, a5, a10 = validate(val_loader, epoch)

        if acc >= best["acc"]:
            best["acc"] = acc
            save_checkpoint({'state_dict': model.state_dict()}, path=save_path)

        if mse <= best["mse"]:
            best["mse"] = mse
            save_checkpoint_mse({'state_dict': model.state_dict()}, path=save_path)

        if hamming < best["hamming"]:
            best["hamming"] = hamming

        for k, v in zip(["a0", "a1", "a2", "a3", "a5", "a10"], [a0, a1, a2, a3, a5, a10]):
            best[k] = max(best[k], v)

        save_checkpoint_time({'state_dict': model.state_dict()}, path=save_path)

        print("##########################################################################################################################")
        print(f"Best Accuracy: {best['acc']:.4f}, Current: {acc:.4f}")
        print(f"Best Hamming Loss: {best['hamming']:.4f}, Current: {hamming:.4f}")
        print(f"Best MSE: {best['mse']:.4f}, Current: {mse:.4f}")
        for k in ['a0', 'a1', 'a2', 'a3', 'a5', 'a10']:
            print(f"Best {k}% acc: {best[k]:.4f}, Current: {locals()[k]:.4f}")
        print("##########################################################################################################################")

    print("Training Complete!")
    print(f"Best Validation Accuracy: {best['acc']:.4f}")

if __name__ == "__main__":
    main()
