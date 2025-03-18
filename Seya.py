import random

responses = {
    "你好": "嗨啦，幹嘛這麼客氣？",
    "天氣": "我又不出門，你自己看窗戶啦！",
    "心情": "我沒心情，你問這幹嘛？"
}
dry_talk = ["別問我啦，我又不是神仙！", "你這問題也太無聊了吧？", "啥？我忙著當機器人，沒空理你！"]

while True:
    user_input = input("你說：")
    found = False
    for key in responses:
        if key in user_input:
            print(responses[key])
            found = True
            break
    if not found:
        print(random.choice(dry_talk))