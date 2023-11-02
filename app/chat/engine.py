import logging
import os
from typing import List

import tiktoken
from langchain.embeddings import XinferenceEmbeddings
from langchain.llms import OpenAI, Xinference
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.callbacks import LlamaDebugHandler
from llama_index.callbacks.base import CallbackManager
from llama_index.chat_engine import CondenseQuestionChatEngine
from llama_index.chat_engine.types import BaseChatEngine
from llama_index.embeddings import LangchainEmbedding
from llama_index.embeddings.openai import (
    OpenAIEmbedding,
    OpenAIEmbeddingMode,
    OpenAIEmbeddingModelType,
)
from llama_index.node_parser import SimpleNodeParser
from llama_index.text_splitter import SentenceSplitter

from ..models.schema import Document as DocumentSchema
from .constants import NODE_PARSER_CHUNK_OVERLAP, NODE_PARSER_CHUNK_SIZE
from .qa_response_synth import get_custom_response_synthesizer, get_template
from .utils import fetch_and_read_documents

logger = logging.getLogger(__name__)


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
            max_tokens=1024,
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


def get_service_context(callback_handlers):
    callback_manager = CallbackManager(callback_handlers)

    embedding_model = get_embedding_model()
    llm = get_llm()

    text_splitter = SentenceSplitter(
        separator=" ",
        chunk_size=NODE_PARSER_CHUNK_SIZE,
        chunk_overlap=NODE_PARSER_CHUNK_OVERLAP,
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


def get_chat_engine(documents: List[DocumentSchema]) -> BaseChatEngine:
    """Custom a query engine for qa, retrieve all documents in one index."""
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)

    service_context = get_service_context([llama_debug])

    llama_index_docs = fetch_and_read_documents(documents)
    logger.debug(llama_index_docs)
    index = VectorStoreIndex.from_documents(
        llama_index_docs,
        service_context=service_context,
    )
    kwargs = {"similarity_top_k": 3}

    query_engine = index.as_query_engine(
        response_synthesizer=get_custom_response_synthesizer(
            service_context=service_context, documents=documents
        ),
        **kwargs,
    )
    chat_engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine,
        condense_question_prompt=get_template(),
        service_context=service_context,
    )
    return chat_engine
