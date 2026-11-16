import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import cv2


# ==================== 1. 特征提取器 ====================
class FeatureExtractor:
    def __init__(self, model, layer_names):
        self.features = {}
        self.hooks = []
        self.layer_names = layer_names
        module_dict = dict(model.named_modules())
        for name in layer_names:
            if name not in module_dict:
                raise ValueError(f"层名 '{name}' 不存在于模型中。请检查 named_modules() 输出。")
            layer = module_dict[name]
            hook = layer.register_forward_hook(self._get_hook(name))
            self.hooks.append(hook)

    def _get_hook(self, name):
        def hook(module, input, output):
            self.features[name] = output.detach().cpu()
        return hook

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ==================== 2. 离散自由能计算 ====================
def compute_free_energy(feature, gamma=0.5):
    phi_sq = feature ** 2
    double_well = 0.25 * (phi_sq - 1.0) ** 2

    grad_x = feature[:, :, :, 2:] - feature[:, :, :, :-2]
    grad_y = feature[:, :, 2:, :] - feature[:, :, :-2, :]

    grad_x = F.pad(grad_x, (1, 1, 0, 0), mode='replicate')
    grad_y = F.pad(grad_y, (0, 0, 1, 1), mode='replicate')

    grad_sq = grad_x ** 2 + grad_y ** 2
    gradient_energy = (gamma / 2.0) * grad_sq

    energy = (double_well + gradient_energy).mean().item()
    return energy


# ==================== 3. 主绘图函数（一次处理 N 张图） ====================
def plot_phase_separation_evidence(model, dataloader, device='cuda',
                                   layer_names=None,
                                   save_dir='./figs',
                                   prefix='phase_sep',
                                   num_samples=10):
    """
    遍历 dataloader，对前 num_samples 个样本各生成 4 张图：
      - {prefix}_{idx}_input.png      输入的有雾图像
      - {prefix}_{idx}_a_distribution.pdf   双峰分布直方图
      - {prefix}_{idx}_b_spatial.pdf        相场空间演化
      - {prefix}_{idx}_c_energy.pdf         能量递减曲线
    """
    os.makedirs(save_dir, exist_ok=True)

    if layer_names is None:
        layer_names = ['Encoder.0', 'Encoder.1', 'Encoder.2']

    model.eval()

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_samples:
            break

        # ---------- 解析 batch ----------
        if isinstance(batch, dict):
            hazy_img = batch['hazy'].to(device)
        elif isinstance(batch, (tuple, list)):
            hazy_img = batch[0].to(device)
        else:
            hazy_img = batch.to(device)

        # ---------- 保存输入图像 ----------
        hazy_np = hazy_img[0].detach().cpu().permute(1, 2, 0).numpy()
        # 如果输入做了 [-1,1] 归一化，转回 [0,1]
        if hazy_np.min() < 0:
            hazy_np = (hazy_np + 1.0) / 2.0
        hazy_np = np.clip(hazy_np, 0, 1)
        path_input = os.path.join(save_dir, f'{prefix}_{batch_idx}_input.png')
        plt.imsave(path_input, hazy_np)
        print(f"[✓] Saved input: {path_input}")

        # ---------- 提取中间特征 ----------
        extractor = FeatureExtractor(model, layer_names)
        with torch.no_grad():
            _ = model(hazy_img)
        features = extractor.features
        extractor.remove_hooks()

        sample_idx = 0

        # -------------------- (a) 双峰分布直方图 --------------------
        fig_a, ax_a = plt.subplots(figsize=(5.5, 4.5))

        deepest_layer = layer_names[-1]
        feat_deep = features[deepest_layer][sample_idx]
        feat_shallow = features[layer_names[0]][sample_idx]

        vals_shallow = feat_shallow.flatten().numpy()
        vals_deep = feat_deep.flatten().numpy()

        bins = np.linspace(-3, 3, 100)
        ax_a.hist(vals_shallow, bins=bins, alpha=0.6, color='gray',
                  label='Before Cahn-AG (shallow)', density=True)
        ax_a.hist(vals_deep, bins=bins, alpha=0.7, color='steelblue',
                  label='After Cahn-AG (deep)', density=True)

        ax_a.axvline(x=1.0, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='Equilibria $\pm 1$')
        ax_a.axvline(x=-1.0, color='r', linestyle='--', linewidth=1.5, alpha=0.7)

        ax_a.set_xlabel('Feature Value (Phase Field $\phi$)', fontsize=11)
        ax_a.set_ylabel('Density', fontsize=11)
        ax_a.legend(fontsize=9)
        ax_a.set_xlim([-3, 3])

        path_a = os.path.join(save_dir, f'{prefix}_{batch_idx}_a_distribution.pdf')
        plt.tight_layout()
        fig_a.savefig(path_a, dpi=300, bbox_inches='tight', format='pdf')
        plt.close(fig_a)
        print(f"[✓] Saved (a): {path_a}")

        # -------------------- (b) 相场空间演化 --------------------
        n_layers = len(layer_names)
        phase_fields = []
        for lname in layer_names:
            feat = features[lname][sample_idx]
            pf = feat.mean(dim=0).numpy()
            pf = np.clip(pf, -1, 1)
            phase_fields.append(pf)

        h, w = phase_fields[0].shape
        gap = 5
        total_width = n_layers * w + (n_layers - 1) * gap
        combined = np.ones((h, total_width)) * 0.5

        for i, pf in enumerate(phase_fields):
            if pf.shape != (h, w):
                pf = cv2.resize(pf, (w, h), interpolation=cv2.INTER_LINEAR)
            start_col = i * (w + gap)
            combined[:, start_col:start_col + w] = pf

        base_height = 3.0
        aspect = total_width / h
        fig_width = base_height * aspect + 1.2
        fig_height = base_height + 0.8

        fig_b, ax_b = plt.subplots(figsize=(fig_width, fig_height))

        im = ax_b.imshow(combined, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax_b.axis('off')

        for i in range(n_layers):
            start_col = i * (w + gap) + w // 2
            ax_b.text(start_col, -20, f'Layer {i + 1}', ha='center', fontsize=9)

        cbar = plt.colorbar(im, ax=ax_b, fraction=0.02, pad=0.01)
        cbar.set_label('$\phi$ (phase field)', fontsize=10)
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels(['Haze\n(-1)', 'Mixed\n(0)', 'Clear\n(+1)'])

        path_b = os.path.join(save_dir, f'{prefix}_{batch_idx}_b_spatial.pdf')
        fig_b.savefig(path_b, dpi=300, bbox_inches='tight', format='pdf')
        plt.close(fig_b)
        print(f"[✓] Saved (b): {path_b}")

        # -------------------- (c) 能量递减曲线 --------------------
        fig_c, ax_c = plt.subplots(figsize=(5.5, 4.5))

        energies = []
        for lname in layer_names:
            feat = features[lname]
            E = compute_free_energy(feat, gamma=0.5)
            energies.append(E)

        x = np.arange(len(layer_names))
        ax_c.plot(x, energies, 'o-', color='steelblue', linewidth=2, markersize=8, label='Cahn-AG')

        ax_c.set_xlabel('Network Depth (Encoder Layer)', fontsize=11)
        ax_c.set_ylabel('Free Energy $E(\phi)$', fontsize=11)
        ax_c.set_xticks(x)
        ax_c.set_xticklabels([f'L{i + 1}' for i in range(len(layer_names))])
        ax_c.legend(fontsize=10)
        ax_c.grid(True, alpha=0.3)

        for i in range(len(energies) - 1):
            if energies[i + 1] < energies[i]:
                ax_c.annotate('', xy=(i + 1, energies[i + 1]), xytext=(i, energies[i]),
                              arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

        path_c = os.path.join(save_dir, f'{prefix}_{batch_idx}_c_energy.pdf')
        plt.tight_layout()
        fig_c.savefig(path_c, dpi=300, bbox_inches='tight', format='pdf')
        plt.close(fig_c)
        print(f"[✓] Saved (c): {path_c}")

    print(f"\n[Done] 共处理 {min(batch_idx + 1, num_samples)} 个样本，结果保存在: {save_dir}")

