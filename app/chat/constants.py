# See https://github.com/run-llama/llama_index/issues/1052
# Set a smaller value to avoid `This model's maximum context length was exceeded`.
NODE_PARSER_CHUNK_SIZE = 128
NODE_PARSER_CHUNK_OVERLAP = 10

DOC_ID_KEY = "doc_id"

ENV_CHAT_HISTORY_KEEP_CNT = "HISTORY_KEEP_CNT"
ENV_LLM_MAX_TOKENS = "LLM_MAX_TOKENS"
