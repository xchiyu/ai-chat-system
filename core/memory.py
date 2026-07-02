import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from llama_index.core.memory import ChatMemoryBuffer


class MemoryManager:
    def __init__(self, token_limit: int = 2048, storage_path: str = "./storage/memory"):
        self.token_limit = token_limit
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self.memory_buffer = ChatMemoryBuffer(token_limit=token_limit)
        self.sessions: List[Dict[str, Any]] = []
        self.current_session_id: Optional[str] = None
        self.load_sessions()

    def load_sessions(self):
        sessions_file = os.path.join(self.storage_path, "sessions.json")
        if os.path.exists(sessions_file):
            try:
                with open(sessions_file, "r", encoding="utf-8") as f:
                    self.sessions = json.load(f)
                print(f"加载了 {len(self.sessions)} 个会话")
            except Exception as e:
                print(f"加载会话失败: {e}")
                self.sessions = []
        else:
            self.sessions = []

    def save_sessions(self):
        sessions_file = os.path.join(self.storage_path, "sessions.json")
        try:
            with open(sessions_file, "w", encoding="utf-8") as f:
                json.dump(self.sessions, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存会话失败: {e}")

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.current_session_id = session_id
        self.sessions.insert(0, {
            "id": session_id,
            "title": "新对话",
            "messages": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        })
        self.save_sessions()
        return session_id

    def get_current_session(self) -> Optional[Dict[str, Any]]:
        if self.current_session_id:
            for session in self.sessions:
                if session["id"] == self.current_session_id:
                    return session
        return None

    def add_message(self, role: str, content: str):
        session = self.get_current_session()
        if not session:
            self.create_session()
            session = self.get_current_session()
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        if session:
            session["messages"].append(message)
            session["updated_at"] = datetime.now().isoformat()
            
            if session["title"] == "新对话" and role == "user":
                session["title"] = content[:30] if len(content) > 30 else content
            
            self.save_sessions()

    def switch_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        for session in self.sessions:
            if session["id"] == session_id:
                self.current_session_id = session_id
                return session
        return None

    def get_sessions_list(self) -> List[Dict[str, Any]]:
        return [{
            "id": session["id"],
            "title": session["title"],
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "message_count": len(session["messages"])
        } for session in self.sessions]

    def get_session_messages(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        for session in self.sessions:
            if session["id"] == session_id:
                return session["messages"]
        return None

    def delete_session(self, session_id: str) -> bool:
        for i, session in enumerate(self.sessions):
            if session["id"] == session_id:
                del self.sessions[i]
                if self.current_session_id == session_id:
                    self.current_session_id = None
                self.save_sessions()
                return True
        return False

    def get_recent_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        session = self.get_current_session()
        if not session:
            return []
        return session["messages"][-limit:]

    def get_context_summary(self, llm) -> str:
        recent_messages = self.get_recent_history(5)
        if not recent_messages:
            return ""
        
        history_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in recent_messages
        ])
        
        prompt = f"""请总结以下对话历史，提取关键信息：

{history_text}

总结："""
        
        try:
            response = llm.complete(prompt)
            return response.text
        except Exception as e:
            print(f"生成上下文摘要失败: {e}")
            return ""

    def reset_memory(self, mode: str = "all"):
        if mode == "all":
            self.sessions = []
            self.current_session_id = None
            self.memory_buffer = ChatMemoryBuffer(token_limit=self.token_limit)
            sessions_file = os.path.join(self.storage_path, "sessions.json")
            if os.path.exists(sessions_file):
                os.remove(sessions_file)
            print("记忆已完全重置")
        elif mode == "recent":
            session = self.get_current_session()
            if session and len(session["messages"]) > 5:
                session["messages"] = session["messages"][:-5]
                self.save_sessions()
            print("最近5条记忆已清除")
        else:
            print("无效的重置模式")

    def get_memory_stats(self) -> Dict[str, Any]:
        total_messages = sum(len(session["messages"]) for session in self.sessions)
        return {
            "total_sessions": len(self.sessions),
            "total_messages": total_messages,
            "current_session_id": self.current_session_id,
            "token_limit": self.token_limit,
            "storage_path": self.storage_path
        }

    def get_memory_buffer(self):
        return self.memory_buffer
