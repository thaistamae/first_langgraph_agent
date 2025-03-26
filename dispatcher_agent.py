import os
from dotenv import load_dotenv
from typing import Dict, Any, List, TypedDict, Optional, Callable, Literal, Annotated, cast, Tuple
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from price_agent import app as price_app, AgentState as PriceAgentState, get_ticker_symbol, search_symbol
from chart_agent import chart_app as chart_app, ChartAgentState

load_dotenv()

class DispatcherState(TypedDict, total=False):
    messages: List[HumanMessage | AIMessage]
    request_type: Optional[str]
    query: str
    ticker_symbol: str
    time_range: Optional[str]
    interval: Optional[str]
    price_result: Optional[Dict[str, Any]]
    chart_result: Optional[Dict[str, Any]]

def detect_request_type(query: str) -> str:
    """
    Detect the type of request from the user query
    
    Returns:
        "price" for current price requests
        "chart" for historical chart requests
    """
    query = query.lower()
    
    chart_keywords = [
        "chart", "graph", "historical", "history", "trend", "trends", "performance", 
        "over time", "last week", "last month", "last year", "past", "movement",
        "plot", "range", "interval"
    ]
    
    for keyword in chart_keywords:
        if keyword in query:
            return "chart"
    
    return "price"

def extract_query_params(query: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Extract ticker symbol, time range, and interval from query
    
    Returns:
        Tuple of (ticker_symbol, time_range, interval)
    """
    query = query.lower()
    ticker_symbol = ""
    time_range = "1mo"  
    interval = "1d"  

    range_mapping = {
        "day": "1d",
        "week": "5d",
        "month": "1mo",
        "3 month": "3mo",
        "three month": "3mo",
        "6 month": "6mo", 
        "six month": "6mo",
        "year": "1y",
        "5 year": "5y",
        "five year": "5y"
    }
    
    for time_desc, range_value in range_mapping.items():
        if time_desc in query:
            time_range = range_value
            break
    
    interval_mapping = {
        "daily": "1d",
        "weekly": "1wk",
        "monthly": "1mo"
    }
    
    for interval_desc, interval_value in interval_mapping.items():
        if interval_desc in query:
            interval = interval_value
            break
    
    words = query.split()
    potential_tickers = []
    
    common_companies = {
        "apple": "AAPL",
        "google": "GOOGL", 
        "microsoft": "MSFT",
        "amazon": "AMZN",
        "tesla": "TSLA",
        "facebook": "META",
        "meta": "META",
        "netflix": "NFLX",
        "nvidia": "NVDA"
    }
    
    for company, ticker in common_companies.items():
        if company in query:
            return ticker, time_range, interval
    
    for word in words:
        if word.isupper() and 1 <= len(word) <= 5 and word.isalpha():
            potential_tickers.append(word)
    
    if potential_tickers:
        return potential_tickers[0], time_range, interval
    
    search_text = " ".join(words[:3])  
    ticker = get_ticker_symbol(search_text)
    
    return ticker, time_range, interval

def classify_request(state: DispatcherState) -> DispatcherState:
    """
    Classify the user request as either price or chart and extract parameters
    """
    state_copy = state.copy()
    
    messages = state_copy.get("messages", [])
    if not messages:
        return state_copy
    
    last_message = messages[-1]
    
    if isinstance(last_message, HumanMessage):
        user_input = last_message.content.strip()
        state_copy["query"] = user_input
        state_copy["request_type"] = detect_request_type(user_input)
        
        ticker, time_range, interval = extract_query_params(user_input)
        state_copy["ticker_symbol"] = ticker
        state_copy["time_range"] = time_range
        state_copy["interval"] = interval
    
    return state_copy

def process_price_request(state: DispatcherState) -> DispatcherState:
    """
    Process a price request using the price agent
    """
    state_copy = state.copy()
    ticker = state_copy.get("ticker_symbol", "")
    
    try:
        price_state: PriceAgentState = {
            "messages": [HumanMessage(content=ticker)],
            "current_symbol": ""
        }
        
        result = price_app.invoke(price_state)
        
        state_copy["price_result"] = result
        
        if result and "messages" in result:
            for message in result["messages"]:
                if isinstance(message, AIMessage):
                    state_copy["messages"].append(message)
    except Exception as e:
        error_msg = f"Error processing price request: {str(e)}"
        state_copy["messages"].append(AIMessage(content=error_msg))
    
    return state_copy

def process_chart_request(state: DispatcherState) -> DispatcherState:
    """
    Process a chart request using the chart agent
    """
    state_copy = state.copy()
    ticker = state_copy.get("ticker_symbol", "")
    time_range = state_copy.get("time_range", "1mo")
    interval = state_copy.get("interval", "1d")
    
    try:
        formatted_query = f"{ticker} range:{time_range} interval:{interval}"
        
        chart_state: ChartAgentState = {
            "messages": [HumanMessage(content=formatted_query)],
            "current_symbol": "",
            "chart_data": None,
            "error": None
        }
        
        result = chart_app.invoke(chart_state)
        
        state_copy["chart_result"] = result
        
        if result and "messages" in result:
            for message in result["messages"]:
                if isinstance(message, AIMessage):
                    state_copy["messages"].append(message)
    except Exception as e:
        error_msg = f"Error processing chart request: {str(e)}"
        state_copy["messages"].append(AIMessage(content=error_msg))
    
    return state_copy

def route_by_request_type(state: DispatcherState) -> str:
    """
    Routes to the appropriate agent based on request type
    """
    request_type = state.get("request_type", "")
    
    if request_type == "price":
        return "process_price"
    else:
        return "process_chart"

def create_dispatcher_workflow():
    workflow = StateGraph(DispatcherState)
    
    workflow.add_node("classify_request", classify_request)
    workflow.add_node("process_price", process_price_request)
    workflow.add_node("process_chart", process_chart_request)
    
    workflow.set_entry_point("classify_request")
    
    workflow.add_conditional_edges(
        "classify_request",
        route_by_request_type,
    )
    
    workflow.add_edge("process_price", END)
    workflow.add_edge("process_chart", END)
    
    return workflow.compile()

dispatcher_app = create_dispatcher_workflow()

if __name__ == "__main__":
    print("Financial Data Agent")
    print("-------------------")
    print("Ask for stock price or historical chart (e.g., 'What's the current price of Apple?' or 'Show me a chart for Tesla')")
    
    user_input = input("> ")
    
    initial_state: DispatcherState = {
        "messages": [HumanMessage(content=user_input)],
        "request_type": None,
        "query": "",
        "ticker_symbol": "",
        "time_range": None,
        "interval": None,
        "price_result": None,
        "chart_result": None
    }
    
    try:
        result = dispatcher_app.invoke(initial_state)
        
        if result and "messages" in result:
            for message in result["messages"]:
                if isinstance(message, AIMessage):
                    print(f"\n{message.content}\n")
        else:
            print("Error: No result returned from the workflow")
    except Exception as e:
        print(f"Error running the workflow: {str(e)}")

__all__ = ['dispatcher_app', 'DispatcherState', 'detect_request_type'] 