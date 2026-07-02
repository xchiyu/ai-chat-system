from openai import OpenAI


#链接到智普平台
API_KEY = "d831f56604634f339e526a8b02bd603a.PVwAtiMLDNEFr6ot"      
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"  
MODEL = "GLM-4"
#新建回话窗口
client = OpenAI(
    api_key=API_KEY, 
    base_url=BASE_URL
    )

#开始对话
#和大模型对话      提示词，prompt
response = client.chat.completions.create(
    model=MODEL,
    #messages 消息列表，包含用户和大模型的对话记录
    #role 角色，user表示用户，assistant表示大模型，system表示系统
    #content 消息内容
    messages=[
        #system 系统提示词，用于引导大模型的行为
        {"role": "system", "content": "你是一个专业的助手"},
        {"role": "user", "content": "你是谁？"}
        ],
    #temperature 温度，取值范围0-1，0表示稳定，1表示活跃
    temperature=0.8,
    #top_p 检索范围
    top_p=0.7
)

print(response.choices[0].message.content)