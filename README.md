---
title: Financial Data Agent
emoji: 📊
colorFrom: purple
colorTo: purple
sdk: gradio
sdk_version: 5.22.0
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Financial Data Agent

A LangGraph-powered application that provides stock information and charts using the Yahoo Finance API.

## Features

- **Unified Interface**: Ask for either current price data or historical charts with natural language
- **Smart Request Routing**: Automatically determines whether you want price data or historical charts
- **Real-time Price Data**: Get current stock prices and key financial metrics
- **Historical Charts**: View stock performance over different time periods with interactive charts
- **Natural Language Understanding**: Parse queries like "What's the current price of Apple?" or "Show me a chart for Tesla over the last 6 months"

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

### Unified Web Interface (Recommended)

For the combined interface that handles both price and chart queries:
```bash
python app.py
```

### Individual Interfaces

For historical charts only:
```bash
python chart_web.py
```

### Command Line Interfaces

Run the price agent from the command line:
```bash
python price_agent.py
```

Run the chart agent from the command line:
```bash
python chart_agent.py
```

Run the dispatcher agent (handles both) from the command line:
```bash
python dispatcher_agent.py
```

## System Components

### Agents

- **Price Agent** (`price_agent.py`): Retrieves current price and financial metrics
- **Chart Agent** (`chart_agent.py`): Creates historical price charts with various timeframes
- **Dispatcher Agent** (`dispatcher_agent.py`): Analyzes user queries and routes to the appropriate agent

### Web Interfaces

- **Price Interface** (`app.py`): Web UI for current stock prices
- **Chart Interface** (`chart_web.py`): Web UI for historical charts
- **Unified Interface** (`app.py`): Combined web UI that handles both types of queries

## Example Queries

- "What's the current price of Apple?"
- "Show me a chart for Tesla over the last 6 months"
- "Microsoft stock price"
- "Amazon historical performance"
- "NVDA"
- "MSFT chart for the past year"

## Data Retrieved

### Price Information
- Current price
- Market cap
- 52-week price range
- P/E ratio
- Dividend yield

### Chart Information
- Historical closing prices
- Customizable time ranges (1d, 5d, 1mo, 3mo, 6mo, 1y, 5y, max)
- Customizable intervals (daily, weekly, monthly)
