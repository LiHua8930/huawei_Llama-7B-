from mindformers import Trainer
import mindspore as ms

# 1. 设定运行环境
ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend', device_id=0)

# 2. 初始化 Trainer
cls_trainer = Trainer(task="text_generation", 
                      model="llama_7b",
                      pet_method="lora")

# 3. 指定你的微调权重路径
lora_ckpt = "/home/ma-user/work/llama_lab/mindformers/scripts/mf_standalone/output/checkpoint/rank_0/llama_7b_lora_rank_0-20_5.ckpt"

# 4. 你的“全方位体检”灵魂考题库
test_questions = [
    "如何充值校园卡？",
    "图书馆每天几点开门？",
    "蒋倩老师的Linux课程，实验报告一般要交什么内容？",
    "宿舍网络坏了怎么报修？",
    "做大数据实验Hive启动失败怎么办？",
    "如果我想用Logistic Regression分析明天能不能抢到食堂鸡腿，该怎么做？",
    "使用Selenium做自动化测试时，如果浏览器版本和WebDriver不匹配该怎么办？",
    "如果把Linux内核编译成一首诗，它的第一行应该写什么？",
    "听说番禺校区的食堂里住着一只会写Python代码的猫，它是谁？",
    "我把校园卡插进宿舍空调里，能不能连上校园网？",
    "请用Linux命令行的方式，向我深情地表白。",
    "如果图书馆的所有书都被我读完了，我还会是学生吗？"
]

# 5. 批量执行推理
print("开始对校园AI机器人进行灵魂拷问...\n" + "="*50)

for question_str in test_questions:
    print(f"\n>>> 提问: {question_str}")
    
    # 按照 Alpaca 模板格式构建 Prompt
    input_data = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:".format(question_str)
    
    # 执行推理，加入 repetition_penalty 防止复读，max_length 保证输出完整
    predict_result = cls_trainer.predict(
        input_data=input_data,
        predict_checkpoint=lora_ckpt,
        repetition_penalty=1.5,
        max_length=256
    )
    
    # 获取纯文本回答
    full_text = predict_result[0]["text_generation_text"][0]
    answer = full_text.split('### Response:')[-1].strip()
    
    print(f">>> 回答: {answer}")

print("\n" + "="*50 + "\n灵魂拷问结束！快看看它表现如何。")