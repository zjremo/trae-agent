import logging
import os

# 日志管理器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# add a StreamHandler to output logs to the terminal
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# Openai测试
from openai import OpenAI

client = OpenAI(base_url="https://api.deepseek.com/v1",
                api_key="###################")

# 创建对话
messages = [
    {
        'role':
        'system',
        'content':
        "You are a chat assistant, your task is to accept the user's request and answer his questions"
    },
    {
        'role': 'user',
        'content': "hello, my name is zjr. Please introduce yourself to me!"
    },
]

# 第一次请求
response = client.chat.completions.create(messages=messages,
                                          model='deepseek-chat',
                                          stream=True,
                                          temperature=0.5,
                                          max_tokens=1000)

# 收集响应并添加到对话历史
full_response = ""
for chunk in response:
    content = chunk.choices[0].delta.content
    # 打字机输出的关键：flush = True， 只要有字符数据就可以向外面吐；不需要等待缓冲区被填满
    print(content, end="", flush=True)  # end参数默认为\n, 但此时设置之后表示不追加任何字符
    full_response += content
print()
messages.append({'role': 'assistant', 'content': full_response})

# 后续对话（示例）
user_input = "评价学术圈草台班子"
messages.append({'role': 'user', 'content': user_input})

logger.info(f"after add context to model, the messages are {messages}")

# 第二次请求（包含上下文）
response = client.chat.completions.create(
    messages=messages,  # 包含完整对话历史
    model='deepseek-chat',
    stream=True,
    temperature=0.5,
    max_tokens=1000)

# 处理新的响应
full_response = ""
for chunk in response:
    content = chunk.choices[0].delta.content
    print(content, end="", flush=True)
    full_response += content
print('')

# # 添加上下文
# messages.append({'role': 'assistant', 'content': full_response})
# logger.info(f"after add context to model, the messages are {messages}")

# user_input = "向我介绍一下python协程编程"
# messages.append({'role': 'assistant', 'content':user_input})
# # 第三次请求（包含上下文）
# response = client.chat.completions.create(messages=messages,
#                                           model='deepseek-chat',
#                                           stream=False,
#                                           temperature=0.5,
#                                           max_tokens=1000)
# print(response.choices[0].message.content)
