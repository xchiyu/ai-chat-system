import os
import time
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from core.config import load_config
from core.models import get_zhipu_llm, get_zhipu_embedding
from core.knowledge import KnowledgeBase
from core.memory import MemoryManager
from core.prompt import PromptGenerator

config = load_config()
zhipu_llm = get_zhipu_llm(config)
zhipu_embedding = get_zhipu_embedding(config)

knowledge_base = KnowledgeBase(
    vector_path=config['system']['storage_path'] + "/vector_store",
    docs_path="./docs/",
    embed_model=zhipu_embedding
)

memory_manager = MemoryManager(storage_path=config['system']['storage_path'] + "/memory")
prompt_generator = PromptGenerator()

app = FastAPI(title="智能对话系统", description="集成智普大模型的对话系统")

os.makedirs("./templates", exist_ok=True)
os.makedirs("./static", exist_ok=True)


class ChatRequest(BaseModel):
    message: str
    use_knowledge: bool = True
    use_memory: bool = True


class ResetRequest(BaseModel):
    mode: str = "all"


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("./templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
async def health():
    return {"status": "ok", "service": "chat_system"}


@app.post("/chat")
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    try:
        start_time = time.time()
        
        if memory_manager.current_session_id is None:
            memory_manager.create_session()
        
        context = ""
        memory_summary = ""
        
        if request.use_knowledge:
            context = knowledge_base.get_relevant_context(request.message)
            
            document_names = knowledge_base.get_document_names()
            for doc_name in document_names:
                if doc_name in request.message or (doc_name.replace('.', '') in request.message):
                    context = knowledge_base.get_all_documents_content()
                    break
        
        if request.use_memory:
            memory_summary = memory_manager.get_context_summary(zhipu_llm)
        
        document_names = knowledge_base.get_document_names()
        doc_names_str = "\n".join([f"- {name}" for name in document_names]) if document_names else "无"
        
        messages = prompt_generator.generate_chat_messages(
            user_query=request.message,
            context=context,
            memory_summary=memory_summary,
            document_names=doc_names_str
        )
        
        response = zhipu_llm.chat(messages)
        response_text = response.message.content
        
        background_tasks.add_task(memory_manager.add_message, "user", request.message)
        background_tasks.add_task(memory_manager.add_message, "assistant", response_text)
        
        response_time = time.time() - start_time
        
        return {
            "response": response_text,
            "context_used": len(context) > 0,
            "memory_used": len(memory_summary) > 0,
            "response_time": round(response_time, 2),
            "session_id": memory_manager.current_session_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/new_session")
async def new_session():
    try:
        session_id = memory_manager.create_session()
        return {"status": "success", "session_id": session_id, "title": "新对话"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions")
async def get_sessions():
    try:
        sessions = memory_manager.get_sessions_list()
        return {"sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    try:
        messages = memory_manager.get_session_messages(session_id)
        if messages is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"messages": messages}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/switch_session/{session_id}")
async def switch_session(session_id: str):
    try:
        session = memory_manager.switch_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"status": "success", "session_id": session_id, "title": session["title"]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    try:
        success = memory_manager.delete_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"status": "success", "message": "会话已删除"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/restore_session")
async def restore_session(request: Dict[str, Any]):
    try:
        session_data = request.get('data')
        if not session_data or 'id' not in session_data:
            raise HTTPException(status_code=400, detail="无效的会话数据")
        
        memory_manager.sessions.insert(0, session_data)
        memory_manager.save_sessions()
        
        if memory_manager.current_session_id is None:
            memory_manager.current_session_id = session_data['id']
        
        return {"status": "success", "message": "会话已恢复"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/restore_sessions")
async def restore_sessions(request: Dict[str, Any]):
    try:
        sessions_data = request.get('sessions', [])
        if not sessions_data:
            raise HTTPException(status_code=400, detail="会话列表为空")
        
        for session_data in reversed(sessions_data):
            if 'id' in session_data:
                existing = next((s for s in memory_manager.sessions if s['id'] == session_data['id']), None)
                if existing:
                    memory_manager.sessions.remove(existing)
                memory_manager.sessions.insert(0, session_data)
        
        memory_manager.save_sessions()
        
        if memory_manager.current_session_id is None and sessions_data:
            memory_manager.current_session_id = sessions_data[0]['id']
        
        return {"status": "success", "message": f"已恢复 {len(sessions_data)} 个会话"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset_memory")
async def reset_memory(request: ResetRequest):
    try:
        if request.mode not in ["all", "recent"]:
            raise HTTPException(status_code=400, detail="无效的重置模式，支持 'all' 和 'recent'")
        
        memory_manager.reset_memory(request.mode)
        return {"status": "success", "message": f"记忆已{request.mode}重置"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory_stats")
async def memory_stats():
    return memory_manager.get_memory_stats()


@app.get("/rebuild_index")
async def rebuild_index():
    try:
        knowledge_base.rebuild_index()
        return {"status": "success", "message": "向量库已重建"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...), rebuild: bool = True):
    try:
        filename = file.filename
        if not filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")
        
        name_without_ext = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1].lower()
        
        if ext not in [".txt", ".docx"]:
            raise HTTPException(status_code=400, detail="不支持的文件格式，仅支持 .txt 和 .docx")
        
        new_filename = f"{name_without_ext}{ext}"
        file_path = os.path.join("./docs/", new_filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        if rebuild:
            knowledge_base.rebuild_index()
        
        return {
            "status": "success",
            "message": f"文档 {new_filename} 上传成功",
            "rebuild_index": rebuild
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/document_list")
async def document_list():
    try:
        docs = []
        for filename in os.listdir("./docs/"):
            file_path = os.path.join("./docs/", filename)
            if os.path.isfile(file_path):
                ext = os.path.splitext(filename)[1].lower()
                if ext in [".txt", ".docx"]:
                    docs.append({
                        "name": filename,
                        "size": os.path.getsize(file_path),
                        "type": ext[1:].upper()
                    })
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete_document/{filename}")
async def delete_document(filename: str):
    try:
        file_path = os.path.join("./docs/", filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="文件不存在")
        
        os.remove(file_path)
        knowledge_base.rebuild_index()
        
        return {"status": "success", "message": f"文档 {filename} 删除成功"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.mount("/static", StaticFiles(directory="./static"), name="static")

if __name__ == "__main__":
    import uvicorn
    port = config['system'].get('server_port', 8088)
    uvicorn.run(app, host="0.0.0.0", port=port)
