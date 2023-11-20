import logging
import os
from typing import List

import tiktoken
from langchain.embeddings import XinferenceEmbeddings
from langchain.llms import Xinference
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.callbacks import LlamaDebugHandler
from llama_index.callbacks.base import CallbackManager
from llama_index.chat_engine.types import BaseChatEngine, ChatMode
from llama_index.embeddings import LangchainEmbedding
from llama_index.embeddings.openai import (
    OpenAIEmbedding,
    OpenAIEmbeddingMode,
    OpenAIEmbeddingModelType,
)
from llama_index.llms import OpenAI
from llama_index.memory import ChatMemoryBuffer
from llama_index.node_parser import SimpleNodeParser
from llama_index.text_splitter import SentenceSplitter

from ..models.schema import Document as DocumentSchema
from .constants import (
    ENV_CHAT_HISTORY_KEEP_CNT,
    ENV_LLM_MAX_TOKENS,
    NODE_PARSER_CHUNK_OVERLAP,
    NODE_PARSER_CHUNK_SIZE,
)
from .qa_response_synth import get_context_prompt_template, get_sys_prompt
from .utils import fetch_and_read_documents

logger = logging.getLogger(__name__)


def get_llm_max_tokens():
    return int(os.environ.get(ENV_LLM_MAX_TOKENS, 1024))


def get_history_count(history_cnt):
    return history_cnt or int(os.environ.get(ENV_CHAT_HISTORY_KEEP_CNT, 10))


def get_llm():
    llm_type = os.getenv("LLM")

    if llm_type == "openai":
        llm = OpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo-0613",
            streaming=False,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif llm_type == "xinference":
        llm = Xinference(
            server_url=os.getenv("XINFERENCE_SERVER_ENDPOINT"),
            model_uid=os.getenv("XINFERENCE_LLM_MODEL_UID"),
            temperature=0.0,
            max_tokens=get_llm_max_tokens(),
        )
    else:
        raise ValueError(f"Unknown LLM type {llm_type}")

    return llm


def get_embedding_model():
    embedding_type = os.getenv("EMBEDDING")

    if embedding_type == "openai":
        embedding = OpenAIEmbedding(
            mode=OpenAIEmbeddingMode.SIMILARITY_MODE,
            model_type=OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    elif embedding_type == "xinference":
        embedding = LangchainEmbedding(
            XinferenceEmbeddings(
                server_url=os.getenv("XINFERENCE_SERVER_ENDPOINT"),
                model_uid=os.getenv("XINFERENCE_EMBEDDING_MODEL_UID"),
            )
        )
    else:
        raise ValueError(f"Unknown EMBEDDING type {embedding_type}")

    return embedding


def get_service_context(callback_handlers, chunk_size, chunk_overlap):
    callback_manager = CallbackManager(callback_handlers)

    embedding_model = get_embedding_model()
    llm = get_llm()

    text_splitter = SentenceSplitter(
        separator=" ",
        chunk_size=chunk_size or NODE_PARSER_CHUNK_SIZE,
        chunk_overlap=chunk_overlap or NODE_PARSER_CHUNK_OVERLAP,
        paragraph_separator="\n\n\n",
        secondary_chunking_regex="[^,.;。]+[,.;。]?",
        tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode,
    )

    node_parser = SimpleNodeParser.from_defaults(
        text_splitter=text_splitter,
        callback_manager=callback_manager,
    )

    return ServiceContext.from_defaults(
        callback_manager=callback_manager,
        llm=llm,
        embed_model=embedding_model,
        node_parser=node_parser,
    )


def get_chat_engine(
    documents: List[DocumentSchema], chunk_size, chunk_overlap, history_cnt
) -> BaseChatEngine:
    """Custom a query engine for qa, retrieve all documents in one index."""
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)

    service_context = get_service_context([llama_debug], chunk_size, chunk_overlap)

    llama_index_docs = fetch_and_read_documents(documents)
    logger.debug(llama_index_docs)
    index = VectorStoreIndex.from_documents(
        llama_index_docs,
        service_context=service_context,
    )

    memory = ChatMemoryBuffer.from_defaults(
        token_limit=get_llm_max_tokens() * get_history_count(history_cnt) * 2
    )

    chat_engine = index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT,
        memory=memory,
        context_template=get_context_prompt_template(documents),
        system_prompt=get_sys_prompt(),
    )
    return chat_engine
