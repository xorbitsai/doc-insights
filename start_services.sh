#!/bin/bash

# Start Streamlit app
poetry run streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0 &

# Keep the script running
wait