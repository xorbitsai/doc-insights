import logging
import os
import pickle
import tempfile

import streamlit as st
from llama_index.llms import ChatMessage, MessageRole

from app.chat.engine import (
    get_chat_engine,
    get_chat_engine_from_persistence,
    get_engine_for_summarization,
)
from app.log import Utf8DecoderFormatter
from app.models.schema import Document, FundDocumentMetadata
from app.chat.constants import (
    NODE_PARSER_CHUNK_OVERLAP,
    NODE_PARSER_CHUNK_SIZE,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

handler = logging.StreamHandler()
handler.setFormatter(Utf8DecoderFormatter())
logger.handlers = []
logger.addHandler(handler)


def init_message_history():
    clear_button = st.sidebar.button("清空对话", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "我是小助手，请问你有什么问题想问？"},
        ]


def init_node_preview():
    if "nodes" in st.session_state:
        with st.expander("文本分块预览"):
            formatted_nodes = [
                {"text": n.text, "strlen": len(n.text), "metadata": n.metadata}
                for n in st.session_state.nodes
            ]
            st.dataframe(formatted_nodes)


def init_engine():
    if "engine" not in st.session_state:
        placeholder = st.empty()
        if os.getenv("PERSISTENCE") != "true":
            _upload_and_index(placeholder)
        else:
            _use_index_or_upload(placeholder)


def _use_index_or_upload(placeholder):
    with placeholder.container():
        use_index_tab, upload_tab = st.tabs(["使用旧的索引", "上传新的文档"])

        with use_index_tab:
            load_index_folder = st.text_input("选择已经保存的索引路径")
            (col,) = st.columns(1)
            col.number_input(
                "保留的历史对话轮数",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                key="history_cnt_with_index",
            )
            click_load = st.button("加载索引，并开启对话", disabled=not bool(load_index_folder))

        with upload_tab:
            st.warning("请先上传本地对话所使用的文档，文档格式为PDF或WORD")
            uploaded_files = st.file_uploader(
                "为本次对话提供相关的文档（可以是多个文档）",
                type=["pdf", "docx", "doc"],
                accept_multiple_files=True,
            )
            index_folder = st.text_input("输入保存索引的路径")

            with st.expander("配置", expanded=True):
                col1, col2, col3 = st.columns(3)

                col1.number_input(
                    "单个文本块最大长度", value=NODE_PARSER_CHUNK_SIZE, key="chunk_size"
                )
                col2.number_input(
                    "相邻文本块重合长度", value=NODE_PARSER_CHUNK_OVERLAP, key="chunk_overlap"
                )
                col3.number_input(
                    "保留的历史对话轮数",
                    min_value=1,
                    max_value=20,
                    value=5,
                    step=1,
                    key="history_cnt",
                )
                st.caption(
                    '一般情况下请保持默认值，如遇到"context length exceeded"的问题，请尝试调小"单个文本块最大长度"'
                )

                click_index = st.button(
                    "添加文件，并开启对话", disabled=len(uploaded_files) == 0 or not index_folder
                )

    if click_load or click_index:
        placeholder.empty()
        if click_load:
            # 使用旧的索引
            with st.spinner("加载索引和初始化，对话即将开始，请耐心等待..."):
                (
                    st.session_state.nodes,
                    st.session_state.engine,
                ) = get_chat_engine_from_persistence(
                    load_index_folder,
                    st.session_state.history_cnt_with_index,
                )
                st.success("索引加载完毕!")

        else:
            # 上传文件
            if not os.path.isdir(index_folder):
                raise Exception("索引目录不是文件夹")
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
                st.session_state.nodes, st.session_state.engine = get_chat_engine(
                    documents,
                    st.session_state.chunk_size,
                    st.session_state.chunk_overlap,
                    st.session_state.history_cnt,
                    persist_dir=os.path.join(index_folder, "storage"),
                )
                with open(os.path.join(index_folder, "documents.pkl"), "wb") as f:
                    pickle.dump(documents, f)
                st.success("索引构建完毕!")


def _upload_and_index(placeholder):
    with placeholder.container():
        st.warning("请先上传本地对话所使用的文档，文档格式为PDF或WORD")
        st.warning("索引不会持久化，下次进入时需要重新上传文件")
        uploaded_files = st.file_uploader(
            "为本次对话提供相关的文档（可以是多个文档）",
            type=["pdf", "docx", "doc"],
            accept_multiple_files=True,
        )

        with st.expander("配置", expanded=True):
            col1, col2, col3 = st.columns(3)

            col1.number_input(
                "单个文本块最大长度", value=NODE_PARSER_CHUNK_SIZE, key="chunk_size"
            )
            col2.number_input(
                "相邻文本块重合长度", value=NODE_PARSER_CHUNK_OVERLAP, key="chunk_overlap"
            )
            col3.number_input(
                "保留的历史对话轮数",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                key="history_cnt",
            )
            st.caption('一般情况下请保持默认值，如遇到"context length exceeded"的问题，请尝试调小"单个文本块最大长度"')

            click = st.button("添加文件，并开启对话", disabled=len(uploaded_files) == 0)

    if click:
        placeholder.empty()
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

            logger.info(f"File {uploaded_file.name} has been written to {file_path}")
        with st.spinner("构建索引和初始化，对话即将开始，请耐心等待..."):
            st.session_state.nodes, st.session_state.engine = get_chat_engine(
                documents,
                st.session_state.chunk_size,
                st.session_state.chunk_overlap,
                st.session_state.history_cnt,
            )
            st.success("索引构建完毕!")


def init_sm_engine():
    if "sm_engine" not in st.session_state:
        placeholder = st.empty()
        with placeholder.container():
            st.warning("请先上传本地对话所使用的文档，文档格式为PDF")
            st.warning("索引不会持久化，下次进入时需要重新上传文件")
            uploaded_file = st.file_uploader("为本次对话提供相关的文档", type=["pdf"])

            with st.expander("配置", expanded=True):
                col1, col2, col3 = st.columns(3)

                col1.number_input(
                    "单个文本块最大长度", value=NODE_PARSER_CHUNK_SIZE, key="chunk_size"
                )
                col2.number_input(
                    "相邻文本块重合长度", value=NODE_PARSER_CHUNK_OVERLAP, key="chunk_overlap"
                )
                st.caption(
                    '一般情况下请保持默认值，如遇到"context length exceeded"的问题，请尝试调小"单个文本块最大长度"'
                )

                click = st.button("添加文件，并开启对话", disabled=uploaded_file is None)

        if click:
            placeholder.empty()
            documents = []
            if uploaded_file is not None:
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
                (
                    st.session_state.nodes,
                    st.session_state.sm_engine,
                ) = get_engine_for_summarization(
                    documents,
                    st.session_state.chunk_size,
                    st.session_state.chunk_overlap,
                )
                st.success("索引构建完毕!")


def kb_chat():
    st.header("知识库对话")

    init_message_history()
    init_engine()
    init_node_preview()

    if "engine" not in st.session_state:
        return

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
                    file_name = source_node.node.metadata["source"]
                    page_label = source_node.node.metadata["page"]

                    shortened_text = f'来源：《{file_name[:25]} ...》"第{page_label}页'
                    with st.expander(shortened_text):
                        st.caption(f"Node id: {node_id}")
                        st.caption(f"source: {file_name}")
                        st.caption(f"Score: {source_node.score}")
                        st.caption(f"Content: {source_node.node.get_content()}")


def sm_chat():
    st.header("总结对话")

    init_message_history()
    init_sm_engine()
    init_node_preview()

    if "sm_engine" not in st.session_state:
        return

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
                engine_response = st.session_state.sm_engine.query(prompt)
                response = str(engine_response.response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
                st.write(response)


if __name__ == "__main__":
    st.set_page_config(page_title="文档问答", page_icon="📚")

    page_names_to_funcs = {"知识库对话": kb_chat, "总结对话": sm_chat}

    demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
    page_names_to_funcs[demo_name]()
