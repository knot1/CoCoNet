import torch
import os
import hydra
from thop import profile

# ===================== 固定配置 =====================
CHECKPOINT_PATH = "/data3/logs/full_model_vlsp34_Vaihingen_42/2026-05-09_09-35-44/results_/data3/logs/full_model_vlsp34_vaihingen/best_model_vaihingen"
WINDOW_SIZE = (256, 256)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 6
IN_CHANS = [3, 1]

# ===================== 加载配置 =====================
def load_hydra_config():
    with hydra.initialize_config_dir(config_dir=os.getcwd(), version_base=None):
        cfg = hydra.compose(config_name="config")
    return cfg.model

# ===================== 构建模型 =====================
def build_model():
    from models.model import Baseline
    model_cfg = load_hydra_config()
    model = Baseline(
        cfg=model_cfg,
        num_classes=NUM_CLASSES,
        in_chans=IN_CHANS,
        class_labels=None
    )
    return model

# ===================== 加载权重 =====================
def load_model(model, weight_path):
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k.replace("module.", "")] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

# ===================== 计算 FLOPs =====================
def calculate_flops(model):
    rgb = torch.randn(1, 3, *WINDOW_SIZE).to(DEVICE)
    dsm = torch.randn(1, 1, *WINDOW_SIZE).to(DEVICE)

    # 包装模型，只取第一个输出（解决多输出报错）
    class WrapModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x1, x2):
            out = self.model(x1, x2)
            return out[0] if isinstance(out, (list, tuple)) else out

    wrapped = WrapModel(model)
    wrapped.eval()

    # 计算
    flops, params = profile(wrapped, inputs=(rgb, dsm), verbose=False)

    # 手动格式化，避开 thop 版本BUG
    gflops = flops / 1e9
    params_m = params / 1e6

    print("\n" + "=" * 70)
    print("📊 模型 GFLOPs & 参数量 计算结果（已修复BUG）")
    print("=" * 70)
    print(f"模型: Baseline")
    print(f"输入尺寸: {WINDOW_SIZE}")
    print(f"参数量: {params_m:.2f} M")
    print(f"计算量: {gflops:.2f} GFLOPs")
    print("=" * 70)

# ===================== 主函数 =====================
if __name__ == "__main__":
    print("✅ 正在构建模型...")
    model = build_model()

    if os.path.exists(CHECKPOINT_PATH):
        print(f"✅ 加载权重: {CHECKPOINT_PATH}")
        model = load_model(model, CHECKPOINT_PATH)
    else:
        print("❌ 权重不存在！")
        exit()

    calculate_flops(model)