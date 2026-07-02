# core/models.py
from typing import Optional, List, Sequence, Dict, Any
from openai import OpenAI
from llama_index.core.llms import CustomLLM, LLMMetadata, ChatMessage, ChatResponse, MessageRole, CompletionResponse
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS

# ---------- 辅助函数 ----------
def to_message_dicts(messages: Sequence[ChatMessage]) -> List[Dict]:
    return [
        {"role": message.role.value, "content": message.content}
        for message in messages
    ]

def get_additional_kwargs(response) -> Dict:
    return {
        "token_counts": response.usage.total_tokens,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }

# ---------- 自定义 LLM ----------
class ChatGLM(CustomLLM):
    num_output: int = DEFAULT_NUM_OUTPUTS
    context_window: int = Field(default=DEFAULT_CONTEXT_WINDOW, description="The maximum number of context tokens for the model.", gt=0)
    model: str = Field(default=None, description="The llm model to use")
    api_key: str = Field(default=None, description="The LLM API key.")
    base_url: str = Field(default=None, description="The LLM Base Url")
    top_p: float = Field(default=0.6, description="Top-p sampling")
    temperature: float = Field(default=0.6, description="Temperature")
    reuse_client: bool = Field(default=True, description="Reuse the client between requests.")

    _client: Optional[Any] = PrivateAttr()

    def __init__(
        self,
        model: str = None,
        reuse_client: bool = True,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        top_p: float = 0.6,
        temperature: float = 0.6,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            reuse_client=reuse_client,
            top_p=top_p,
            temperature=temperature,
            **kwargs,
        )
        self._client = None

    def _get_client(self) -> OpenAI:
        if not self.reuse_client:
            return OpenAI(api_key=self.api_key, base_url=self.base_url)
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    @classmethod
    def class_name(cls) -> str:
        return "chatglm_llm"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model,
        )

    def _chat(self, messages: List[Dict], stream=False) -> Any:
        response = self._get_client().chat.completions.create(
            model=self.model,
            messages=messages,
            top_p=self.top_p,
            temperature=self.temperature,
            stream=stream,
        )
        return response

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        message_dicts = to_message_dicts(messages)
        response = self._chat(message_dicts, stream=False)
        return ChatResponse(
            message=ChatMessage(
                content=response.choices[0].message.content,
                role=MessageRole(response.choices[0].message.role),
                additional_kwargs={},
            ),
            raw=response,
            additional_kwargs=get_additional_kwargs(response),
        )

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any):
        raise NotImplementedError("Stream chat not implemented.")

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        messages = [{"role": "user", "content": prompt}]
        response = self._chat(messages, stream=False)
        return CompletionResponse(
            text=response.choices[0].message.content,
            raw=response,
            additional_kwargs=get_additional_kwargs(response),
        )

    def stream_complete(self, prompt: str, **kwargs: Any):
        raise NotImplementedError("Stream complete not implemented.")

# ---------- 自定义 Embedding ----------
class ChatGLMEmbeddings(BaseEmbedding):
    model: str = Field(default='embedding-3', description="The embedding model to use.")
    api_key: str = Field(default=None, description="The API key.")
    base_url: str = Field(default=None, description="The Base Url")
    reuse_client: bool = Field(default=True, description="Reuse the client.")

    _client: Optional[Any] = PrivateAttr()

    def __init__(
        self,
        model: str = 'embedding-3',
        reuse_client: bool = True,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            reuse_client=reuse_client,
            **kwargs,
        )
        self._client = None

    def _get_client(self) -> OpenAI:
        if not self.reuse_client:
            return OpenAI(api_key=self.api_key, base_url=self.base_url)
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    @classmethod
    def class_name(cls) -> str:
        return "ChatGLMEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        response = self._get_client().embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            embeddings.append(self._get_text_embedding(text))
        return embeddings

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)

# ---------- 工厂函数 ----------
def get_zhipu_llm(config):
    llm_cfg = config['llm']
    return ChatGLM(
        model=llm_cfg['model'],
        api_key=llm_cfg['api_key'],
        base_url=llm_cfg['api_base'],
        temperature=llm_cfg.get('temperature', 0.6),
        top_p=llm_cfg.get('top_p', 0.6),
        reuse_client=True,
    )

def get_zhipu_embedding(config):
    embed_cfg = config['embedding']
    return ChatGLMEmbeddings(
        model=embed_cfg['model'],
        api_key=embed_cfg['api_key'],
        base_url=embed_cfg['api_base'],
        reuse_client=True,
    )