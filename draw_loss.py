import re
import os

import matplotlib.pyplot as plt

def parse_log(log_path, start_epoch=2000):
    epoch_losses = []
    current_epoch = None
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 查找Epoch号
            epoch_match = re.search(r'Epoch (\d+):', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
            # 查找loss，支持科学计数法和小数
            loss_match = re.search(r'loss:\s*([0-9.eE+-]+)', line)
            if loss_match and current_epoch is not None and current_epoch >= start_epoch:
                try:
                    loss = float(loss_match.group(1))
                    epoch_losses.append((current_epoch, loss))
                except ValueError:
                    continue
    return epoch_losses

def plot_loss(epoch_losses, save_path='loss_curve.png'):
    if not epoch_losses:
        print("No loss data found.")
        return
    epochs, losses = zip(*epoch_losses)
    plt.figure(figsize=(10,5))
    plt.plot(epochs, losses, marker='o', linewidth=1, markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    log_file = 'RWT1x1027_5e5_2.log'  # 替换为你的log文件路径
    start_epoch = 400000              # 可修改起始epoch
    epoch_losses = parse_log(log_file, start_epoch)
    base_name = os.path.splitext(log_file)[0]
    png_file = f"{base_name}_from{start_epoch}.png"
    plot_loss(epoch_losses, save_path=png_file)
