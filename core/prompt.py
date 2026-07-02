from typing import List, Dict, Any, Optional
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate


class PromptGenerator:
    def __init__(self):
        self.default_system_prompt = """你是一个专业的智能助手，擅长根据提供的文档内容精准回答问题。

你拥有以下知识库文档：
{document_names}

请严格遵循以下回答原则：
1. 精准定位：只提取与问题直接相关的信息
2. 简洁精炼：用最少的文字回答问题，避免冗长和多余解释
3. 直击要点：直接给出答案，不重复问题内容，不添加无关信息
4. 长度控制：回答不超过100字，通常一句话即可
5. 格式要求：不使用列表、不使用引号、不使用特殊符号
6. 语言规范：使用中文回答，表达清晰
7. 无信息时：明确说明"未找到相关信息"

回答示例：
- 用户问"张小娟是什么专业的？" → 回答"计算机科学专业"
- 用户问"test里面有什么？" → 回答"关于张小娟的个人信息，包括她的专业、年级和梦想"
- 用户问"今天天气怎么样？" → 回答"未找到相关信息"（如果文档中没有天气信息）"""

    def generate_prompt(
        self,
        user_query: str,
        context: str = "",
        memory_summary: str = "",
        system_prompt: Optional[str] = None
    ) -> str:
        messages = []
        
        messages.append(ChatMessage(
            role=MessageRole.SYSTEM,
            content=system_prompt or self.default_system_prompt
        ))
        
        if memory_summary:
            messages.append(ChatMessage(
                role=MessageRole.SYSTEM,
                content=f"对话历史摘要：{memory_summary}"
            ))
        
        if context:
            messages.append(ChatMessage(
                role=MessageRole.SYSTEM,
                content=f"参考上下文：\n{context}"
            ))
        
        messages.append(ChatMessage(
            role=MessageRole.USER,
            content=user_query
        ))
        
        chat_template = ChatPromptTemplate(messages)
        return chat_template.format()

    def generate_chat_messages(
        self,
        user_query: str,
        context: str = "",
        memory_summary: str = "",
        system_prompt: Optional[str] = None,
        document_names: str = ""
    ) -> List[ChatMessage]:
        messages = []
        
        base_prompt = system_prompt or self.default_system_prompt
        if "{document_names}" in base_prompt and document_names:
            base_prompt = base_prompt.replace("{document_names}", document_names)
        elif "{document_names}" in base_prompt:
            base_prompt = base_prompt.replace("{document_names}", "无")
        
        messages.append(ChatMessage(
            role=MessageRole.SYSTEM,
            content=base_prompt
        ))
        
        if memory_summary:
            messages.append(ChatMessage(
                role=MessageRole.SYSTEM,
                content=f"对话历史摘要：{memory_summary}"
            ))
        
        if context:
            messages.append(ChatMessage(
                role=MessageRole.SYSTEM,
                content=f"参考上下文：\n{context}"
            ))
        
        messages.append(ChatMessage(
            role=MessageRole.USER,
            content=user_query
        ))
        
        return messages

    def set_system_prompt(self, prompt: str):
        self.default_system_prompt = prompt

    def get_default_system_prompt(self) -> str:
        return self.default_system_prompt
