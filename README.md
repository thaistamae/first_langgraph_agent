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

3. You'll need to sign up for a RapidAPI key at https://rapidapi.com to use the Yahoo Finance API.

## Usage

### Command Line Interface

Run the agent from the command line:
```bash
python main.py
```

### Web Interface

For a nicer web-based interface, run:
```bash
python app.py
```

This will start a local web server with a Gradio interface. Open the URL displayed in the console (typically http://127.0.0.1:7860) in your web browser.

## Features

- Query company information by ticker symbol (e.g., AAPL) or company name (e.g., Apple)
- Retrieves real-time stock price data
- Shows key financial metrics including:
  - Current price
  - Market cap
  - 52-week price range
  - P/E ratio
  - Dividend yield
- Smart company name to ticker symbol conversion
