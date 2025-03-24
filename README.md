---
title: First Langgraph Agent
emoji: 📊
colorFrom: purple
colorTo: purple
sdk: gradio
sdk_version: 5.22.0
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Company Info Agent

A simple LangGraph agent that consults company information using the Yahoo Finance API.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the root directory and add your RapidAPI key:
```
RAPIDAPI_KEY=your_api_key_here
```

3. Run the agent:
```bash
python main.py
```

## Usage

The agent can be queried with company symbols (e.g., AAPL for Apple, GOOGL for Google) to get basic company information.

## Note

You'll need to sign up for a RapidAPI key at https://rapidapi.com to use the Yahoo Finance API.
