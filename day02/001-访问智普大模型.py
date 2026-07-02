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

result=client.embeddings.create(
    model="Embedding-3",
    input="你好"
)
print(len(result.data[0].embedding))
