import json
from transformers import AutoTokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling

# 1. 讀取並處理資料集 (假設資料集是 JSONL 格式)
def load_jsonl(file_path):
    dialogues = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())  # 每行解析為 JSON 格式
            dialogues.append({"text": f"問：{item['input']} 回答：{item['output']}"} )
    return dialogues

# 讀取 JSONL 格式資料集
dialogues = load_jsonl(r'C:\VScode\AI老婆\資料集\sample_500k.jsonl')

# 2. 將對話數據轉換為 Hugging Face 的 Dataset 格式
dataset = Dataset.from_list(dialogues)

# 3. 加載Tokenizer 和 預訓練模型 (更換為 DialoGPT-medium)
model_name = "microsoft/DialoGPT-medium"  # 使用 DialoGPT-medium
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 設定 pad_token 為 eos_token
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_name)

# 4. 編碼資料
def encode(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=128)

# 編碼資料集
dataset = dataset.map(encode, batched=True)

# 5. 創建資料處理器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT-2 不是 MLM (Masked Language Model)
)

# 6. 訓練參數設定
training_args = TrainingArguments(
    output_dir="./DialoGPT_finetuned",  # 模型保存的路徑
    overwrite_output_dir=True,
    num_train_epochs=3,  # 訓練輪次
    per_device_train_batch_size=4,  # 批量大小
    save_steps=10_000,  # 保存模型的步數
    save_total_limit=2,  # 保留的模型數量
    logging_dir="./logs",  # 日誌記錄位置
    fp16=True,  # 啟用混合精度訓練
)

# 7. 使用Trainer API進行訓練
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# 開始訓練
trainer.train()

# 8. 保存訓練好的模型和tokenizer
model.save_pretrained("./DialoGPT_finetuned")
tokenizer.save_pretrained("./DialoGPT_finetuned")


