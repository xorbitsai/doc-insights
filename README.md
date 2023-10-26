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

## Run the app

In you want to run a local dev environment, the following command will let you test the application with OpenAI API.

```bash
poetry install
LLM=openai EMBEDDING=openai streamlit run app/main.py
```

## Troubleshooting

* If you want to use OpenAI, check that you've created an .env file that contains your valid (and working) API keys.

