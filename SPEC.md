## 2. 模型架構：LAVTOne + Learnable Prompt

### 2.1 整體概念

採用 `lavt_one` 架構，BERT 內建在模型中。原本 `lavt_one` 的 `text_encoder` 接收外部傳入的 tokenized text 和 attention mask；改為模型內部自動生成 learnable prompt，不再從外部接收任何文字輸入。

模型的 forward 簽名從 `forward(x, text, l_mask)` 簡化為 `forward(x)`。圖片進去、segmentation mask 出來，language feature 的產生完全在模型內部完成。

### 2.2 Learnable Prompt 設計

在 `LAVTOne` 內部，用固定文字模板 + learnable soft tokens 組成 prompt，通過內建的 BERT 產生 language feature。

- **模板結構**：`[CLS] a slice of chest ct with [V1][V2]...[Vn] lung nodule [SEP]`
- `[V1]~[Vn]` 是 learnable `nn.Parameter`，維度為 `(n_soft_tokens, 768)`。
- 固定文字部分（prefix: "a slice of chest ct with"、suffix: "lung nodule"）在 `__init__` 時預先 tokenize，將 token ids 存為 buffer。
- Forward 時：取固定 token 的 word embedding → 和 soft tokens 拼接 → 加上 position embedding 和 token_type embedding → 過 BERT LayerNorm → 過 BERT encoder → 得到 language feature。
- 最終 language feature 的 shape 為 `(B, 768, L)`，`l_mask` 為全 1 的 `(B, L, 1)`，與原本 fusion module 的接口完全相容。

### 2.3 BERT 訓練策略

與原版 `lavt_one` 一致：BERT 全部參數加入 optimizer 一起訓練，不做任何凍結。`bert.pooler` 設為 None。

### 2.4 Soft Token 初始化

用 BERT encode 完整句子 "a slice of chest ct with abnormal lung nodule"，取 soft token 對應位置的 hidden state 作為初始值。這比 random init 收斂更快更穩定。

### 2.5 Soft Token 數量

由 `--n_soft_tokens` 控制，預設 4，合理範圍 4~8。

---

## 5. Dataset 修改

### 5.1 回傳值簡化

`__getitem__` 只回傳 `(image, target)`。

- `image`：經過 transform 的圖片 tensor。
- `target`：binary mask tensor (H, W)，值為 0 或 1。
- 當 `mask == "empty"` 時，回傳全零 target。

不再回傳 `sentences`、`attentions`、`meta`。

### 5.2 Dynamic Negative Resampling

- `__init__` 時分開儲存正樣本 list 和負樣本候選 pool（掃描 images 目錄中不在正樣本清單的 slice）。
- 提供 `resample_negatives(epoch)` 方法，每個 epoch 從候選 pool 隨機抽取負樣本，數量 = `len(正樣本) × neg_ratio`。
- 保留 `PatientAwareBatchSampler`，只用 `patient_id` 分組，不涉及 category。

---

## 6. Loss 函數

### 6.1 判斷邏輯

按 per-sample 的 `target.sum()` 判斷：有前景像素 = 正樣本，全零 = 負樣本。

### 6.2 正樣本

Focal loss + Dice loss（跟原本一樣）。

### 6.3 負樣本

只算 Focal loss，不算 Dice loss（Dice 在 target 全零時分母趨近零，數值不穩定）。負樣本 loss 乘以 0.5 權重，避免壓制 sensitivity。

## 7. Training Loop 修改

### 7.1 train_one_epoch

- Dataloader 只吐 `(image, target)`。
- 直接 `seg_out = model(image)`，不傳文字、category、is_pos。
- `loss = criterion(seg_out, target)`。
- 不再需要外部 `bert_model`。
- 移除 `class_embed_contrastive_loss` 的加入（原本 epoch >= 5 時加入）。

### 7.2 evaluate

- `seg_out = model(image)`。
- 正樣本（target 有前景像素）：計算 IoU。
- 負樣本（target 全零）：檢查 prediction 是否也全零，統計 TN rate。
- 輸出 metrics：Mean IoU、Overall IoU、TN rate。

---

## 8. Optimizer 參數分組

### 8.1 新的參數組

與原版 `lavt_one` 風格一致，BERT 全部參數加入 optimizer：

| 參數組 | Learning Rate | Weight Decay |
|--------|---------------|--------------|
| Backbone norm / pos_embed | base lr | 0.0 |
| Backbone 其餘參數 | base lr | 預設 |
| SimpleDecoding classifier | base lr | 預設 |
| Soft tokens | base lr | 預設 |
| BERT encoder layer 0~9 | base lr | 預設 |

Soft tokens 和 BERT 層統一用 base lr，與原版 lavt_one 對 text_encoder 的處理方式一致。

## 9. CLI 參數修改

### 9.2 新增

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--n_soft_tokens` | `4` | learnable prompt token 數量 |

### 9.3 保留

- `--ck_bert`：用於 model 內部初始化 BERT 權重
- `--neg_ratio`：預設改為 `0.5`，控制 normal slice 數量

---