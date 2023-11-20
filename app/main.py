import logging
import os
import tempfile

import streamlit as st
from llama_index.llms import ChatMessage, MessageRole

from app.chat.engine import get_chat_engine
from app.log import Utf8DecoderFormatter
from app.models.schema import Document, FundDocumentMetadata

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

handler = logging.StreamHandler()
handler.setFormatter(Utf8DecoderFormatter())
logger.handlers = []
logger.addHandler(handler)


def init_page():
    st.set_page_config(page_title="知识库问答助手", page_icon="🤗")
    st.header("知识库问答助手")
    st.sidebar.title("Options")
    st.warning("请先上传本地对话所使用的文档，文档格式为PDF")
    st.warning("索引不会持久化，下次进入时需要重新上传文件")
    st.text_input(
        '切分文本的块大小，如果遇到"context length exceeded"的问题，请尝试调小此值', 256, key="chunk_size"
    )
    st.text_input("切分文本时每个分块的重叠量，一般情况下请保持默认值", 10, key="chunk_overlap")
    st.number_input(
        "保留的历史对话轮数", min_value=1, max_value=20, value=5, step=1, key="history_cnt"
    )


def init_message_history():
    clear_button = st.sidebar.button("清空对话", key="clear")
    if clear_button or "messages" not in st.session_state:
        # reset chat engine
        if "engine" in st.session_state:
            st.session_state.engine.reset()
        st.session_state.messages = [
            {"role": "assistant", "content": "我是小助手，请问你有什么问题想问？"},
        ]


def handle_uploaded_file():
    placeholder = st.empty()
    uploaded_files = placeholder.file_uploader(
        "为本次对话提供相关的文档（可以是多个PDF文档）", type=["pdf"], accept_multiple_files=True
    )
    if len(uploaded_files) > 0:
        placeholder.empty()
        st.success("文件上传成功!")
        return uploaded_files
    else:
        st.stop()


def init_engine():
    if "engine" not in st.session_state:
        if uploaded_files := handle_uploaded_file():
            documents = []
            for uploaded_file in uploaded_files:
                temp_dir = tempfile.mkdtemp()
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                documents.append(
                    Document(
                        url=file_path,
                        metadata=FundDocumentMetadata(
                            document_description=uploaded_file.name
                        ),
                    )
                )

                logger.info(
                    f"File {uploaded_file.name} has been written to {file_path}"
                )
            with st.spinner("构建索引和初始化，对话即将开始，请耐心等待..."):
                st.session_state.engine = get_chat_engine(
                    documents,
                    st.session_state.chunk_size,
                    st.session_state.chunk_overlap,
                    st.session_state.history_cnt,
                )
                st.success("索引构建完毕!")


def main():
    init_page()
    init_message_history()
    init_engine()

    history = []
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        st.chat_message(role).write(content)
        history.append(
            ChatMessage(
                role=MessageRole.USER if role == "user" else MessageRole.ASSISTANT,
                content=content,
            )
        )

    if prompt := st.chat_input(placeholder=f"请问我任何问题"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("请稍等..."):
                engine_response = st.session_state.engine.chat(
                    prompt,
                    # chat_history=history
                )
                response = str(engine_response.response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
                st.write(response)

                for source_node in engine_response.source_nodes:
                    node_id = source_node.node.node_id or "None"
                    file_name = source_node.node.metadata["file_name"] or "None"
                    page_label = source_node.node.metadata["page_label"] or "None"

                    shortened_text = f'来源：《{file_name[:25]} ...》"第{page_label}页'
                    with st.expander(shortened_text):
                        st.caption(f"Node id: {node_id}")
                        st.caption(f"File: {file_name}")
                        st.caption(f"Score: {source_node.score}")
                        st.caption(f"Content: {source_node.node.get_content()}")


if __name__ == "__main__":
    main()
