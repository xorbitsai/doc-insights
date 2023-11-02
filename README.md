# doc-insights
## How it works

- Create a chat engine with [LlamaIndex](https://www.llamaindex.ai/) to answer question based on a set of pre-selected documents.
- Leverage [Streamlit](https://streamlit.io/) for file uploads and interactive communication with the engine.

## Deployment

1. Clone the repo

2. You can run the docker-compose command to launch the app with docker containers, and then type a question in the chat interface.

```bash
docker-compose up --build
```

### Integration with Xinference
1. Start Xinference cluster
```shell
xinference --log-level debug
```

2. Launch an Embedding model and a LLM model, get their model_uids. For example, 
launching ``bge-large-zh`` (embedding) and ``chatglm3`` (LLM):
```python
from xinference.client import Client
client = Client("http://127.0.0.1:9997")
model_uid = client.launch_model(model_name="bge-large-zh", model_type="embedding")
model_uid2 = client.launch_model(model_name="chatglm3", quantization=None, model_format='pytorch', model_size_in_billions=6)
print(model_uid, model_uid2)
```

3. Modify ``docker-compose.yml`` using the above model_uids, for example:
```yaml
version: "2"

services:

  app:
    build: .
    network_mode: "host"
    ports:
      - "8501:8501"
    volumes:
      - ./app:/app/app
    environment:
      - LLM=xinference
      - EMBEDDING=xinference
      - XINFERENCE_SERVER_ENDPOINT=http://127.0.0.1:9997
      - XINFERENCE_EMBEDDING_MODEL_UID=<model_uid>
      - XINFERENCE_LLM_MODEL_UID=<model_uid2>
      - HISTORY_KEEP_CNT=10
```

4. Deploy this application:
```shell
docker-compose up --build
```

## Run the app

In you want to run a local dev environment, the following command will let you test the application with OpenAI API.

```bash
poetry install
LLM=openai EMBEDDING=openai streamlit run app/main.py
```

## Troubleshooting

* If you want to use OpenAI, check that you've created an .env file that contains your valid (and working) API keys.

