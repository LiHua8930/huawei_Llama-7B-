from mindformers.pipeline import pipeline

# 1. 加载模型
pipeline_task = pipeline(task="text_generation", model="llama_7b", max_length=1300)

# do_sample=False 贪婪搜索
pipeline_result = pipeline_task("晴天风铃还有加油站里的其他小伙伴都是超级善良可爱的人呢~", do_sample=False)

print(pipeline_result)