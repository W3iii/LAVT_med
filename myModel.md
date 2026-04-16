# Lung Nodule VLM Segmentation — Research Roadmap

> 基於對話討論整理，針對實驗室資料集（1600 cases，10000 張 2D slice，4 類結節）的研究規劃。

---

## 研究目標

**最終目標**：3D text-conditioned lung nodule segmentation，投 CVPR/ICCV。

**核心 Novelty**：把 text conditioning 帶進 3D nodule segmentation，目前幾乎沒有人做。

**Reviewer 需要回答的問題**：「為什麼要用 text？text 帶來了什麼純 image model 做不到的？」

**回答**：Open-vocabulary segmentation，訓練時只看過 cls1~cls4，但 inference 時給任意形態描述也能分割。

---

## 資料集

### 實驗室資料集
- **規模**：1600 patients，10000 張 2D slice，原始為 3D npy
- **類別**：對應 Lung-RADS v2022
  - cls1 → Category 2（Benign）
  - cls2 → Category 3（Probably Benign）
  - cls3 → Category 4A（Suspicious）
  - cls4 → Category 4B + 4X（Very Suspicious）
- **Mask**：binary，每個類別獨立一張
- **結節大小（2D）**：
  | 類別 | Median | 問題 |
  |---|---|---|
  | cls1 benign | 3 px | 極小，模型難學 |
  | cls2 prob_benign | 10 px | 偏小 |
  | cls3 prob_suspicious | 29 px | 合理 |
  | cls4 suspicious | 108 px | 最大 |

### Text Descriptions（基於 Lung-RADS v2022）

結節類型包含 solid / part-solid / GGN，sentences 需要分別設計：

```python
SENTENCES = {
    # cls1: Lung-RADS 2 — solid < 6mm
    1: [
        "benign solid pulmonary nodule smaller than six millimeters "
        "with smooth margin and regular round shape",
        "small solid pulmonary nodule smaller than six millimeters "
        "with well-circumscribed smooth border",
        "benign appearing solid nodule less than six millimeters "
        "with regular oval shape and no aggressive features",
    ],
    # cls2: Lung-RADS 3 — solid 6-8mm
    2: [
        "probably benign solid pulmonary nodule with well-defined border "
        "and solid diameter between six and eight millimeters",
        "probably benign solid nodule six to eight millimeters "
        "with smooth to mildly irregular border requiring follow-up",
        "indeterminate solid nodule between six and eight millimeters "
        "with no spiculation or pleural involvement",
    ],
    # cls3: Lung-RADS 4A — solid 8-15mm
    3: [
        "suspicious solid pulmonary nodule with irregular margin "
        "and solid diameter between eight and fifteen millimeters",
        "suspicious solid nodule eight to fifteen millimeters "
        "with irregular border requiring three month follow-up",
        "probably malignant solid nodule between eight and fifteen millimeters "
        "with lobulated or irregular margin",
    ],
    # cls4: Lung-RADS 4B/4X — solid ≥ 15mm
    4: [
        "highly suspicious solid pulmonary nodule larger than fifteen millimeters "
        "with spiculated border and irregular shape",
        "very suspicious solid nodule fifteen millimeters or larger "
        "with spiculated margin and possible pleural retraction",
        "highly suspicious solid pulmonary nodule larger than fifteen millimeters "
        "with spiculated border requiring tissue sampling",
    ],
}
```

> **注意**：資料集包含 solid / part-solid / GGN 三種類型，若 annotation 有 nodule type 資訊，建議為每種類型分別設計 sentences（共 4×3 = 12 種組合），確保 text-image 對應精確。

---

## 現階段：2D Prototype（LAVT）

### 目的
驗證 text-conditioned segmentation 的概念，為 3D 鋪路。

### 架構
```
CT slice (480×480, grayscale → 3ch)
    ↓
Swin-B (patch=4, window=12) + PWAM × 4
    ↓
BioBERT text embedding + class embedding (concat)
    ↓
SimpleDecoding (FPN-style decoder)
    ↓
Binary mask (2-class output)
```

### Class Embedding 設計

```python
# model 內
self.class_embed    = nn.Embedding(5, 768)   # 0=neg, 1~4=cls1~4
self.class_pos_embed = nn.Parameter(torch.zeros(1, 768, 1))
nn.init.normal_(self.class_embed.weight, std=0.02)
nn.init.normal_(self.class_pos_embed, std=0.02)

# forward
cls_emb  = self.class_embed(category).unsqueeze(-1)  # (B, 768, 1)
cls_emb  = cls_emb + self.class_pos_embed
l_new    = torch.cat([text_emb, cls_emb], dim=-1)    # (B, 768, seq+1)
cls_mask = torch.ones(B, 1, 1, device=l_mask.device)
l_mask_new = torch.cat([l_mask, cls_mask], dim=1)    # (B, seq+1, 1)
```

Concat 優於相加的原因：PWAM 的 attention 可以選擇性地 attend 到 class token，當 text embedding 區分性不足時（cosine > 0.93），模型自然更依賴 class token。

### Loss 設計

```python
CLS_LOSS_WEIGHT = {0: 1.0, 1: 2.0, 2: 1.5, 3: 1.0, 4: 1.0}

_WEIGHT_TABLE = torch.tensor(
    [CLS_LOSS_WEIGHT.get(i, 1.0) for i in range(5)],
    dtype=torch.float32
)

def criterion(seg_out, target, is_pos, category=None):
    has_fg = target.flatten(1).sum(1) > 0

    if not has_fg.any():
        return torch.tensor(0.0, device=seg_out.device)

    pos_input  = seg_out[has_fg]
    pos_target = target[has_fg]

    bce_per   = F.cross_entropy(pos_input, pos_target,
                                reduction='none').mean(dim=(1,2))
    dice_per  = dice_loss_per_sample(pos_input, pos_target)
    focal_per = focal_loss_per_sample(pos_input, pos_target,
                                      alpha=0.75, gamma=2.0)
    loss_per  = bce_per + dice_per + focal_per

    if category is not None:
        pos_cat  = category[has_fg]
        cls_w    = _WEIGHT_TABLE.to(seg_out.device)[pos_cat]
        seg_loss = (loss_per * cls_w).sum() / cls_w.sum()
    else:
        seg_loss = loss_per.mean()

    return seg_loss
```

**設計理由**：
- 負樣本不算 seg loss，讓 class embedding 負責區分（而非 exist head）
- per-sample dice + focal 確保 cls_w 正確作用到每個 loss component
- 向量化，無 Python for loop

### Contrastive Loss（輔助）

```python
def contrastive_loss_embed(class_embed_weight, temperature=0.07):
    embs = F.normalize(class_embed_weight, dim=-1)  # (4, 768)
    sim  = torch.matmul(embs, embs.T) / temperature # (4, 4)
    mask = ~torch.eye(4, dtype=torch.bool, device=embs.device)
    return sim[mask].mean()

# train loop
class_embed_weight = model.class_embed.weight[1:5]
total_loss = seg_loss + 0.1 * contrastive_loss_embed(class_embed_weight)
```

### Sampler

`PatientAwareBatchSampler`：round-robin interleave，確保同 batch 不同 patient，避免 overfit 到 patient-specific 特徵。

### 訓練設定

```bash
python train_ln.py \
    --model lavt \
    --model_id lavt_ln \
    --ln_dataset_root ../dataset \
    --batch-size 32 \
    --lr 0.00005 \
    --wd 1e-2 \
    --swin_type base \
    --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
    --window12 \
    --epochs 100 \
    --img_size 480 \
    --neg_ratio 0.5 \
    --workers 4 \
    --pin_mem \
    --output-dir ./checkpoints/ln
```

### 已知問題與對策

| 問題 | 原因 | 對策 |
|---|---|---|
| cls1 IoU 低（~10%） | median 3px，Swin patch=4 downsampling 後消失 | class weight=2.0；長期換 CNN encoder |
| TN rate 低（~50%） | class embedding 未完整訓練前模型分不清四類 | class embedding + contrastive loss |
| precision@0.9 = 0% | 邊界不精確 | 未來加 boundary loss 或 deep supervision |

---

## 中期：2D 優化方向

### 1. Deep Supervision（對小結節最有效）

在 decoder 每層加 auxiliary loss，淺層 feature map 解析度高，對小結節更友善：

```python
loss = criterion(out_full,    target) * 1.0 \
     + criterion(out_half,    F.interpolate(target)) * 0.5 \
     + criterion(out_quarter, F.interpolate(target)) * 0.25
```

參考：nnU-Net 的 deep supervision 設計。

### 2. Text Encoder 升級

將 BERT 換成 **BioBERT**（`dmis-lab/biobert-base-cased-v1.2`）：

```python
# 只改兩行
bert_model = BertModel.from_pretrained('dmis-lab/biobert-base-cased-v1.2')
self.tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2')
```

BioBERT 對醫療術語（spiculated, ground-glass, part-solid）的 embedding 更準確。

### 3. nnU-Net Preprocessing

```python
def nnunet_style_normalize(volume, clip_lower, clip_upper, mean, std):
    volume = np.clip(volume, clip_lower, clip_upper)
    volume = (volume - mean) / (std + 1e-8)
    return volume

# 用資料集統計取代 ImageNet mean/std
# clip_lower ≈ -1000 HU, clip_upper ≈ 400 HU
```

---

## 長期：3D Text-conditioned Segmentation

### 架構：TextConditioned3DUNet

```
3D patch (64×64×64, 以結節中心 crop)
    ↓
DecomposedConvBlock encoder × 4
    ↓
AxialTextAttention × 4（PWAM slice-wise + z-axis attention）
    ↓
Decoder with skip connections × 4
    ↓
Binary mask (2-class, per class prompt)
```

### DecomposedConvBlock

```python
class DecomposedConvBlock(nn.Module):
    """
    3D conv 分解成 spatial(1×3×3) + depth(3×1×1)
    參數量比標準 3D conv 少，對小結節友善
    參考：Text3DSAM (CVPR 2025 Workshop)
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01),
        )
        self.depth = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, kernel_size=(3,1,1), padding=(1,0,0)),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01),
        )
        self.residual = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.depth(self.spatial(x)) + self.residual(x)
```

### AxialTextAttention

解決「3D flatten 破壞 z 軸空間關係」的問題，分兩步注入 text：

```python
class AxialTextAttention(nn.Module):
    """
    Step 1: Slice-wise PWAM（每個 z slice 獨立做 text cross-attention）
    Step 2: Z-axis attention（沿 z 軸做 text cross-attention，保留結節連續性）

    參考：
    - Axial-DeepLab (ECCV 2020)：分軸 attention
    - TGSAM-2 (MICCAI 2025)：z 軸 text tracking
    """
    def __init__(self, feat_ch, text_dim=768):
        super().__init__()
        self.slice_attn = PWAM(feat_ch, text_dim)   # 直接重用現有 PWAM
        self.z_attn     = nn.MultiheadAttention(feat_ch, num_heads=8, batch_first=True)
        self.z_norm     = nn.LayerNorm(feat_ch)
        self.text_proj  = nn.Linear(text_dim, feat_ch)

    def forward(self, x, text_emb, l_mask):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape

        # Step 1: slice-wise（xy 平面）
        x_2d   = x.permute(0,2,1,3,4).reshape(B*D, C, H, W)
        t_rep  = text_emb.unsqueeze(1).expand(-1,D,-1,-1).reshape(B*D, C, -1)
        m_rep  = l_mask.unsqueeze(1).expand(-1,D,-1,-1).reshape(B*D, -1, 1)
        x_flat = x_2d.flatten(2).permute(0,2,1)
        x_flat = self.slice_attn(x_flat, t_rep, m_rep)
        x      = x_flat.permute(0,2,1).reshape(B*D, C, H, W)\
                       .reshape(B, D, C, H, W).permute(0,2,1,3,4)

        # Step 2: z-axis（跨 slice）
        x_z    = x.permute(0,3,4,2,1).reshape(B*H*W, D, C)
        t_kv   = self.text_proj(text_emb.permute(0,2,1))
        t_kv   = t_kv.unsqueeze(1).expand(-1,H*W,-1,-1).reshape(B*H*W, -1, C)
        x_z, _ = self.z_attn(x_z, t_kv, t_kv)
        x_z    = self.z_norm(x_z + x.permute(0,3,4,2,1).reshape(B*H*W, D, C))
        x      = x_z.reshape(B, H, W, D, C).permute(0,4,3,1,2)

        return x
```

### 3D Patch Sampling 策略

```python
# 每個結節產生 4 筆訓練資料
for nodule in volume.nodules:
    center = nodule.center_voxel
    patch  = crop_3d(volume, center, size=64)

    # 正樣本（1筆）
    yield {'patch': patch, 'mask': nodule.mask_3d,
           'prompt': SENTENCES[nodule.category],
           'class': nodule.category, 'is_pos': 1}

    # 負樣本（3筆，錯誤的 class prompt）
    for wrong_cat in [1,2,3,4]:
        if wrong_cat != nodule.category:
            yield {'patch': patch, 'mask': zeros_3d,
                   'prompt': SENTENCES[wrong_cat],
                   'class': wrong_cat, 'is_pos': 0}
```

> **重點**：不需要完全沒有結節的 patch。負樣本是「同一個有結節的 patch + 錯誤的 class prompt」，模型必須靠 class embedding 和 text 區分。

### 3D Inference 策略

**短期**：Sliding Window

```python
# patch_size=64, stride=32
# 每個 window × 4 個 prompt = ~5000 次 forward（慢但可跑通）
```

**長期**：Two-stage Pipeline

```
Stage 1: Detection（CPM-Net 或 nnDetection）
  整個 volume → 候選結節中心座標（< 10 個）
      ↓
Stage 2: Text-conditioned Segmentation
  以候選座標 crop 64×64×64 patch
  → TextConditioned3DUNet
  → 4 個 prompt 各跑一次
  → exist head 過濾
  → 輸出精細 3D mask
```

Stage 2 只需要 < 40 次 forward pass，推論快。

---

## 時間規劃

```
Phase 1（現在，1~2 個月）
  ├── 完成 class embedding + contrastive loss
  ├── 2D 結果穩定（Mean IoU > 35%）
  └── 確認 text conditioning 對四類有區分效果

Phase 2（2~3 個月）
  ├── 設計 TextConditioned3DUNet
  ├── 以 2D 的 loss / class embedding 設計為基礎
  ├── Decomposed conv encoder + AxialTextAttention
  └── 2D → 3D patch 資料集轉換

Phase 3（3~4 個月）
  ├── 3D 訓練與調優
  ├── Sliding window inference 實作
  ├── LIDC-IDRI cross-dataset 實驗（open-vocabulary 驗證）
  └── 整理論文
```

---

## 論文實驗設計建議

### 核心實驗

**1. Ablation Study**

| 設定 | Mean IoU | TN rate |
|---|---|---|
| Baseline（無 text） | - | - |
| + BERT text | - | - |
| + BioBERT text | - | - |
| + class embedding | - | - |
| + contrastive loss | - | - |
| Full model | - | - |

**2. Open-vocabulary 實驗**

```
訓練：cls1, cls2, cls3（不看 cls4）
測試：給 "suspicious nodule with spiculated border" prompt
→ 模型能不能分割出 cls4？
```

如果可以，就證明了 text conditioning 的泛化價值。

**3. Cross-dataset 實驗**

```
在實驗室資料集訓練
在 LIDC-IDRI 測試（或反過來）
→ 證明泛化能力
```

---

## 參考文獻

| 論文 | 貢獻 | 相關性 |
|---|---|---|
| LAVT (CVPR 2022) | Language-Aware Vision Transformer | 現在用的 backbone |
| TGSAM-2 (MICCAI 2025) | SAM2 + text conditioning for 3D CT | z-axis text tracking 概念 |
| Text3DSAM (CVPR 2025 Workshop) | 3D text-guided segmentation | decomposed conv 概念 |
| Axial-DeepLab (ECCV 2020) | 分軸 attention | z-axis attention 設計根據 |
| CoOp (IJCV 2022) | Learnable prompt token | class embedding 設計根據 |
| nnU-Net (Nature Methods 2021) | 自動化 3D 醫療分割 | preprocessing + deep supervision |
| Lung-RADS v2022 (ACR) | 臨床分類標準 | text description 設計根據 |
| LIDC-IDRI | 形態學標注資料集 | cross-dataset 實驗用 |