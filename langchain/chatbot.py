import gradio as gr
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_ollama import ChatOllama

# 初始化 LLM 和对话链
model = ChatOllama(
    model="deepseek-r1:8b",
    temperature=0.75
)

store = {}
session_id = "default_user"

def get_session_history(session_id: str):
    """使用内存存储对话历史"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

chain = prompt | model

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

def clear():
    """清空聊天历史记录"""
    if session_id in store:
        store[session_id].clear()
    return None

def send(message, history):
    """处理用户消息并流式返回回复"""
    stream = chain_with_history.stream(
        {"question": message},
        config={"configurable": {"session_id": session_id}}
    )

    response = ""
    for chunk in stream:
        response += chunk.content
        yield response


# 启动 Gradio 界面
chatbot = gr.Chatbot(label="聊天记录")
interface = gr.ChatInterface(
    send,
    title="LocalChatBot",
    description="基于Ollama的本地聊天机器人",
    chatbot=chatbot
)

with interface:
    clear_btn = gr.Button("清除对话历史")
    clear_btn.click(clear, None, interface.chatbot, queue=False)

interface.launch(server_name="0.0.0.0", server_port=7860)
