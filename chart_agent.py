import os
from dotenv import load_dotenv
import requests
import json
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, List, TypedDict, Optional
from langgraph.graph import Graph, END
from langchain_core.messages import HumanMessage, AIMessage
from main import get_ticker_symbol

load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
YAHOO_FINANCE_CHART_URL = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/v3/get-chart"

headers = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": "apidojo-yahoo-finance-v1.p.rapidapi.com"
}

class ChartAgentState(TypedDict):
    messages: List[HumanMessage | AIMessage]
    current_symbol: str
    chart_data: Optional[Dict[str, Any]]
    error: Optional[str]

def fetch_chart_data(symbol: str, interval: str = "1d", range: str = "1mo") -> Optional[Dict[str, Any]]:
    """
    Fetch historical chart data for a company ticker symbol
    
    Args:
        symbol: The company ticker symbol (e.g., AAPL)
        interval: Data interval (e.g., 1d, 1wk, 1mo)
        range: Time range for data (e.g., 1d, 5d, 1mo, 3mo, 6mo, 1y, 5y, max)
    
    Returns:
        Dictionary containing chart data or None if an error occurs
    """
    querystring = {
        "symbol": symbol,
        "interval": interval,
        "range": range,
        "region": "US"
    }
    
    try:
        response = requests.get(
            YAHOO_FINANCE_CHART_URL, 
            headers=headers, 
            params=querystring
        )
        
        response.raise_for_status()
        
        data = response.json()
        
        if "chart" not in data:
            print(f"Error: Unexpected response format - 'chart' key missing")
            return None
            
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"Request error: {str(e)}")
        return None
    except ValueError as e:
        print(f"JSON parsing error: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None

def create_chart_html(data: Dict[str, Any], symbol: str, time_range: str) -> Optional[str]:
    """
    Create an HTML representation of the chart from the data
    
    Args:
        data: Chart data from Yahoo Finance API
        symbol: The ticker symbol
        time_range: The time range used for the chart
        
    Returns:
        HTML string of the chart or None if an error occurs
    """
    try:
        chart_data = data.get("chart", {})
        result = chart_data.get("result", [])
        
        if not result or len(result) == 0:
            print(f"No chart data available for {symbol}")
            return None
        
        timestamps = result[0].get("timestamp", [])
        quotes = result[0].get("indicators", {}).get("quote", [])
        
        if not timestamps or not quotes or len(quotes) == 0:
            print(f"Incomplete data received for {symbol}")
            return None
        
        close_prices = quotes[0].get("close", [])
        
        if not close_prices:
            print(f"No price data available for {symbol}")
            return None
        
        import datetime
        dates = [datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d') for ts in timestamps]
        
        df = pd.DataFrame({
            'Date': dates,
            'Close': close_prices
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name=symbol))
        
        fig.update_layout(
            title=f"{symbol} Stock Price Chart ({time_range})",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_white"
        )
        
        chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        return chart_html
        
    except Exception as e:
        print(f"Error creating chart: {str(e)}")
        return None

def process_chart_query(state: ChartAgentState) -> ChartAgentState:
    """
    Process the user query and fetch chart data
    """
    messages = state.get("messages", [])
    if not messages:
        return state
        
    last_message = messages[-1]
    
    if isinstance(last_message, HumanMessage):
        user_input = last_message.content.strip()
        
        range_param = "1mo"
        interval_param = "1d"
        
        if "range:" in user_input.lower():
            parts = user_input.lower().split("range:")
            if len(parts) > 1:
                range_part = parts[1].strip()
                for r in ["1d", "5d", "1mo", "3mo", "6mo", "1y", "5y", "max"]:
                    if r in range_part.split():
                        range_param = r
                        break
        
        if "interval:" in user_input.lower():
            parts = user_input.lower().split("interval:")
            if len(parts) > 1:
                interval_part = parts[1].strip()
                for i in ["1d", "1wk", "1mo"]:
                    if i in interval_part.split():
                        interval_param = i
                        break
        
        query = user_input.lower()
        for param in [f"range:{range_param}", f"interval:{interval_param}"]:
            query = query.replace(param, "").strip()
        
        symbol = get_ticker_symbol(query)
        state["current_symbol"] = symbol
        
        try:
            chart_data = fetch_chart_data(symbol, interval=interval_param, range=range_param)
            
            if not chart_data:
                error_msg = f"Could not fetch chart data for '{query}' ({symbol}). Please try a different company name or ticker symbol."
                state["messages"].append(AIMessage(content=error_msg))
                state["error"] = error_msg
                return state
            
            state["chart_data"] = chart_data
            
            chart_html = create_chart_html(chart_data, symbol, range_param)
            
            if not chart_html:
                error_msg = f"Could not create chart for '{query}' ({symbol}). The data may be incomplete."
                state["messages"].append(AIMessage(content=error_msg))
                state["error"] = error_msg
                return state
            
            response = f"""
            ## Historical Stock Chart for {symbol}
            
            Time Range: {range_param}
            Interval: {interval_param}
            
            {chart_html}
            """
            
            state["messages"].append(AIMessage(content=response))
            
        except Exception as e:
            error_msg = f"Error processing request for '{query}': {str(e)}"
            state["messages"].append(AIMessage(content=error_msg))
            state["error"] = error_msg
    
    return state

chart_workflow = Graph()

chart_workflow.add_node("process_chart_query", process_chart_query)

chart_workflow.set_entry_point("process_chart_query")

chart_workflow.add_edge("process_chart_query", END)

chart_app = chart_workflow.compile()

if __name__ == "__main__":
    symbol = "AAPL"
    data = fetch_chart_data(symbol)
    
    if data:
        print(f"Successfully retrieved chart data for {symbol}")
        result = data.get("chart", {}).get("result", [])
        if result and len(result) > 0:
            timestamps = result[0].get("timestamp", [])
            quotes = result[0].get("indicators", {}).get("quote", [])
            
            if timestamps and quotes and len(quotes) > 0:
                print(f"Data points: {len(timestamps)}")
                print(f"First timestamp: {timestamps[0]}")
                print(f"Sample quote data: {quotes[0].get('close', [])[0] if quotes[0].get('close') else 'N/A'}")
    else:
        print(f"Failed to retrieve chart data for {symbol}")

__all__ = ['chart_app', 'ChartAgentState', 'fetch_chart_data', 'create_chart_html'] 