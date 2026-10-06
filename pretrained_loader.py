import torch
from text.phonemize.symbols import symbols
from models import SynthesizerTrn, MultiPeriodDiscriminator
import utils


def load_pretrained_mb_istft_vits_G(checkpoint_path, config_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    # mb_istft_vitsの保存形式に合わせて取り出す
    if 'model' in ckpt:
        pretrained_dict = ckpt['model']
    else:
        pretrained_dict = ckpt

    hps = utils.get_hparams_from_file(config_path)
    model = SynthesizerTrn(
      len(symbols), 
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      **hps.model
    )

    model_dict = model.state_dict()
    matched, skipped_shape, skipped_missing = [], [], []
    new_state_dict = {}

    for k, v in model_dict.items():
        if k in pretrained_dict:
            if pretrained_dict[k].size() == v.size():
                new_state_dict[k] = pretrained_dict[k]
                matched.append(k)
            else:
                # 元重みより重みのsizeが増えている場合、その差ぶんの重みを疑似生成し末尾に加える
                new_state_dict[k] = v 
                skipped_shape.append(f"{k}: ckpt={pretrained_dict[k].size()} vs model={v.size()}")

                hidden_dim = v.shape[1]
                mean_val = v.mean().item()
                std_val = v.std().item()
                print("pretrained_dict[k]: ", pretrained_dict[k].size())

                additional_weight = torch.randn(
                    (v.shape[0]-pretrained_dict[k].shape[0], hidden_dim), 
                    dtype=torch.bfloat16, 
                ) * std_val + mean_val
                print("additional_weight: ", additional_weight.size())

                # 元モデルのembed_tokensとadditional_weightを結合しサイズを拡張
                fixed_embed_weight = torch.cat([pretrained_dict[k], additional_weight], dim=0)
                print("fixed_embed_weight: ", fixed_embed_weight.size())
                      
                new_state_dict[k] = fixed_embed_weight
                matched.append(fixed_embed_weight)
        else:
            new_state_dict[k] = v  # ランダム初期化のまま
            skipped_missing.append(k)

    model.load_state_dict(new_state_dict)
    print(f"\n流用成功: {len(matched)} layers")
    print(f"スキップ: {len(skipped_shape)} layers")

    for s in skipped_shape:
        print(f"   {s}")

    print(f"新規初期化: {len(skipped_missing)} layers")
    
    for s in skipped_missing:
        print(f"   {s}")
    return model, ckpt


def load_pretrained_mb_istft_vits_D(checkpoint_path, config_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    # mb_istft_vitsの保存形式に合わせて取り出す
    if 'model' in ckpt:
        pretrained_dict = ckpt['model']
    else:
        pretrained_dict = ckpt

    hps = utils.get_hparams_from_file(config_path)
    model = MultiPeriodDiscriminator(hps.model.use_spectral_norm)

    model_dict = model.state_dict()
    matched, skipped_shape, skipped_missing = [], [], []
    new_state_dict = {}

    for k, v in model_dict.items():
        if k in pretrained_dict:
            if pretrained_dict[k].size() == v.size():
                new_state_dict[k] = pretrained_dict[k]
                matched.append(k)
            else:
                # 元重みより重みのsizeが増えている場合、その差ぶんの重みを疑似生成し末尾に加える
                new_state_dict[k] = v 
                skipped_shape.append(f"{k}: ckpt={pretrained_dict[k].size()} vs model={v.size()}")

                """hidden_dim = v.shape[1]
                mean_val = v.mean().item()
                std_val = v.std().item()
                print("pretrained_dict[k]: ", pretrained_dict[k].size())

                additional_weight = torch.randn(
                    (v.shape[0]-pretrained_dict[k].shape[0], hidden_dim), 
                    dtype=torch.bfloat16, 
                ) * std_val + mean_val
                print("additional_weight: ", additional_weight.size())

                # 元モデルのembed_tokensとadditional_weightを結合しサイズを拡張
                fixed_embed_weight = torch.cat([pretrained_dict[k], additional_weight], dim=0)
                print("fixed_embed_weight: ", fixed_embed_weight.size())
                      
                new_state_dict[k] = fixed_embed_weight
                matched.append(fixed_embed_weight)"""
        else:
            new_state_dict[k] = v  # ランダム初期化のまま
            skipped_missing.append(k)

    model.load_state_dict(new_state_dict)
    print(f"\n流用成功: {len(matched)} layers")
    print(f"スキップ: {len(skipped_shape)} layers")

    for s in skipped_shape:
        print(f"   {s}")

    print(f"新規初期化: {len(skipped_missing)} layers")
    
    for s in skipped_missing:
        print(f"   {s}")
    return model, ckpt


# model G
"""model, ckpt = load_pretrained_mb_istft_vits_G(
    config_path="./checkpoints/jsut_44100_+2_uprate.json",
    checkpoint_path="./checkpoints/G_0.pth"
)
checkpoint = {
    'iteration': ckpt['iteration'], 
    'model': model.state_dict(),
    'learning_rate': ckpt['learning_rate'],
}
torch.save(checkpoint, "./checkpoints/G_0_re.pth")"""

# model D
model, ckpt = load_pretrained_mb_istft_vits_D(
    config_path="./checkpoints/jsut_44100_+2_uprate.json",
    checkpoint_path="./checkpoints/D_0.pth"
)
checkpoint = {
    'iteration': ckpt['iteration'], 
    'model': model.state_dict(),
}
torch.save(checkpoint, "./checkpoints/D_0_re.pth")
print("saved!")