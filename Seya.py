import random
import datetime
import sys
# 確保程式輸出使用 UTF-8 編碼
sys.stdout.reconfigure(encoding='utf-8')

# 載入中文GPT-2模型和Tokenizer
dialo_model = AutoModelForCausalLM.from_pretrained("ckiplab/gpt2-base-chinese")
dialo_tokenizer = AutoTokenizer.from_pretrained("ckiplab/gpt2-base-chinese")
dialo_generator = pipeline("text-generation", model=dialo_model, tokenizer=dialo_tokenizer, 
                           top_k=50, top_p=0.95, temperature=0.9, truncation=True)
# 預設回應，這些回應是固定的，不隨機
responses = {
    "你好": "嗨呀，我是Seya！有什麼我可以幫忙的嗎？",
    "天氣": "我又不出門，你自己看窗戶啦！",
    "早安": "早安呀！今天感覺如何？",
    "晚上好": "晚上好呀，今天過得怎麼樣？"
}

# 幫助辨識是否為常見的問題（如查詢時間、天氣等）
def handle_simple_queries(user_input):
    if "幾點" in user_input or "時間" in user_input:
        # 隨機敷衍回應
        return random.choice([
            "現在是大概這個時候啦，自己猜吧！",
            f"呃…現在應該是{datetime.datetime.now().hour}點{datetime.datetime.now().minute}分吧。",
            "我怎麼知道，時間一直在走啊！"
        ])
    elif "天氣" in user_input:
        return random.choice([
            "我又不出門，天氣怎麼樣你自己看窗戶啦！",
            "天氣？我哪知道，自己查吧。",
            "天氣那種東西，你不如自己看一眼外面！"
        ])
    return None

# 不耐煩回應
dry_talk = ["別問我啦，我又不是神仙！", "我忙著補妝，沒空理你！", "這問題太簡單了，我不想回答！"]

# 開始對話
print("Seya: 嗨呀，我是Seya！有什麼我可以幫忙的嗎？")

while True:
    user_input = input("你說：")
    
    # 如果用戶輸入退出，結束對話
    if user_input.lower() == "退出":
        print("Seya: 好的，期待下次與你再見！")
        break
    
    # 先檢查是否為簡單的問題（時間、天氣等）
    simple_response = handle_simple_queries(user_input)
    if simple_response:
        print(f"Seya: {simple_response}")
    else:
        # 若沒有預設回應，生成對話回應
        ai_response = dialo_generator(f"用中文回答：{user_input}", max_length=150, num_return_sequences=1)
        generated_text = ai_response[0]['generated_text']
        
        # 去掉多餘的提示文字
        generated_text = generated_text.replace("用中文回答：", "").strip()
        
        # 檢查是否為不耐煩的回應（簡單問題或其他）
        if "補妝" in generated_text or "神仙" in generated_text:
            print(f"Seya: {random.choice(dry_talk)}")
        else:
            print(f"Seya: {generated_text}")
