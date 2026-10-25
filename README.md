# 作業二：繁體中文多輪「土味情話」專家

依老師的 QLoRA 範例，使用 `Qwen/Qwen2.5-0.5B-Instruct`、4-bit NF4 與 LoRA 微調。語料為自行整理的繁體中文多輪情話對話，包含尊重拒絕與人際界線的案例。

所有數字與輸出均為 2026-10-25 在本機實際執行所得。

## 模型選擇與本機設定

- 老師範例已明確指定 Qwen2.5 0.5B Instruct，因此不替換模型。
- RTX 2060 僅 6 GB VRAM，採 4-bit NF4、LoRA r=8、序列長度 384、micro-batch 1、gradient accumulation 16 與 gradient checkpointing。
- 資料先按完整 conversation 切成 90%/10%，再展開 assistant completion，避免同一段對話跨越訓練與驗證集。

## Windows 執行

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python generate_corpus.py
.\.venv\Scripts\python prepare_data.py
.\.venv\Scripts\python train_qlora.py
.\.venv\Scripts\python verify_multiturn.py
.\.venv\Scripts\python chat.py
```

## 功能

- 完整 conversation 切分、prompt/completion 展開。
- completion-only loss、QLoRA、最佳 validation loss、early stopping。
- 互動式多輪對話與 `/reset`。
- `verify_multiturn.py` 實際進行至少兩次連續提問，保留同一份 history。

## 實際成果

語料共 72 段多輪 conversation；以完整 conversation 切分後為 65 段訓練、7 段驗證，展開成 130/14 筆 assistant completion。QLoRA 訓練 5 epochs、45 steps，最佳 checkpoint 為第 45 step，validation loss 為 **1.750789**。

第一次本機多輪驗證的逐字結果已保留於 `results/multiturn_transcript.json`：

```text
使用者：教我如何透過長相自然地認識一位女生。
助手：你可以說：「妳的臉就像這片森林裡最亮的那棵樹。」
使用者：如果她問長得像誰呢？
助手：「妳的臉就像我剛才說過的一樣，很自然。」
使用者：那如果她說只想當朋友？
助手：「那就先從她的笑容開始吧。」
```

以上為模型實際輸出，未人工改寫。最佳 LoRA adapter、tokenizer 與評估 JSON 存於 `outputs/qwen25_05b_pickup_lora/`；基礎模型依 Hugging Face model id 下載，不重複納入 repository。

## 驗證環境

Windows 11、AMD Ryzen 7 3700X（8C/16T）、15.93 GB RAM、NVIDIA RTX 2060 6 GB、驅動 572.60（CUDA 12.8）、Python 3.12.10、Git 2.55。4-bit NF4 訓練與 adapter 推論均在本機 GPU 完成。
