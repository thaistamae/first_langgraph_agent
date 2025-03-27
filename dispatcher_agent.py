import os
from dotenv import load_dotenv
from typing import Dict, Any, List, TypedDict, Optional, Callable, Literal, Annotated, cast, Tuple
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from price_agent import app as price_app, AgentState as PriceAgentState, get_ticker_symbol, search_symbol
from chart_agent import chart_app as chart_app, ChartAgentState

load_dotenv()

# Initialize HuggingFace model
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",  # More powerful instruction model
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1,
    huggingfacehub_api_token=HF_TOKEN,
)

class DispatcherState(TypedDict, total=False):
    messages: List[HumanMessage | AIMessage]
    request_type: Optional[str]
    query: str
    ticker_symbol: str
    time_range: Optional[str]
    interval: Optional[str]
    price_result: Optional[Dict[str, Any]]
    chart_result: Optional[Dict[str, Any]]

# System prompt template for financial data extraction
SYSTEM_PROMPT = """
You are a financial assistant specialized in understanding stock queries.

When a user asks about stock information, carefully analyze their query to extract:

1. The company name or ticker symbol mentioned
2. Whether they want current price information or historical chart data
3. Any time range mentioned (day, week, month, 3 months, 6 months, year, 5 years)
4. Any interval preference (daily, weekly, monthly)

For well-known companies, you know their ticker symbols:
- Apple = AAPL
- Microsoft = MSFT
- Amazon = AMZN
- Google/Alphabet = GOOGL
- Meta/Facebook = META
- Tesla = TSLA
- Netflix = NFLX
- Nvidia = NVDA
- ServiceNow = NOW
- Salesforce = CRM
- IBM = IBM
- Oracle = ORCL
- CrowdStrike = CRWD
- AMD = AMD
- Intel = INTC

Your job is to parse the query: "{query}"

Respond with a JSON object containing exactly these fields:
{{
  "request_type": "price" or "chart",
  "ticker": the ticker symbol (if you can determine it) or company name (if you're not sure of ticker),
  "time_range": preferred time range code or "1mo" if not specified,
  "interval": preferred interval code or "1d" if not specified
}}

Time range codes: "1d" (day), "5d" (week), "1mo" (month), "3mo" (3 months), "6mo" (6 months), "1y" (year), "5y" (5 years)
Interval codes: "1d" (daily), "1wk" (weekly), "1mo" (monthly)

If the user mentions historical data, trends, charts, or graphs, classify as "chart", otherwise "price".
"""

def extract_financial_data(query: str) -> Dict[str, Any]:
    """
    Extract financial data from query using the LLM
    
    Returns:
        Dictionary with request_type, ticker, time_range, and interval
    """
    try:
        print(f"[DEBUG] Extracting financial data from: '{query}'")
        
        # Format the prompt with the user query
        prompt = SYSTEM_PROMPT.format(query=query)
        
        # Get LLM response
        response = llm.invoke(prompt)
        print(f"[DEBUG] LLM raw response: {response}")
        
        # Extract JSON from response
        import json
        import re
        
        # First, try to find JSON in the response
        json_match = re.search(r'({.*})', response.replace('\n', ' '), re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            # Handle potential issues with JSON formatting
            json_str = json_str.replace("'", '"')
            try:
                result = json.loads(json_str)
                print(f"[DEBUG] Parsed result: {result}")
                return result
            except json.JSONDecodeError:
                print(f"[DEBUG] JSON parsing error with: {json_str}")
        
        # If JSON extraction fails, try a more robust approach
        print("[DEBUG] Falling back to regex extraction")
        
        # Extract request_type
        request_type = "price"
        if re.search(r'"request_type"\s*:\s*"chart"', response):
            request_type = "chart"
        
        # Extract ticker
        ticker_match = re.search(r'"ticker"\s*:\s*"([^"]+)"', response)
        ticker = ticker_match.group(1) if ticker_match else ""
        
        # Extract time_range
        time_range_match = re.search(r'"time_range"\s*:\s*"([^"]+)"', response)
        time_range = time_range_match.group(1) if time_range_match else "1mo"
        
        # Extract interval
        interval_match = re.search(r'"interval"\s*:\s*"([^"]+)"', response)
        interval = interval_match.group(1) if interval_match else "1d"
        
        result = {
            "request_type": request_type,
            "ticker": ticker,
            "time_range": time_range,
            "interval": interval
        }
        
        print(f"[DEBUG] Regex extracted result: {result}")
        return result
        
    except Exception as e:
        print(f"[DEBUG] Error extracting financial data: {str(e)}")
        # Return defaults if extraction fails
        return {
            "request_type": "price",
            "ticker": "",
            "time_range": "1mo",
            "interval": "1d"
        }

def get_stock_ticker(company_name: str) -> str:
    """
    Get the stock ticker from a company name
    
    Returns:
        Stock ticker if found, otherwise empty string
    """
    print(f"[DEBUG] Looking up ticker for: '{company_name}'")
    
    # Check if the company name looks like a ticker already (all caps, 1-5 chars)
    if company_name.isupper() and 1 <= len(company_name) <= 5 and company_name.isalpha():
        print(f"[DEBUG] Company name '{company_name}' looks like a ticker already")
        return company_name
    
    # Try to get the ticker using the search_symbol function
    ticker = search_symbol(company_name)
    if ticker:
        print(f"[DEBUG] Found ticker '{ticker}' for '{company_name}' via search")
        return ticker
    
    # If that fails, try the more direct get_ticker_symbol approach
    ticker = get_ticker_symbol(company_name)
    if ticker:
        print(f"[DEBUG] Found ticker '{ticker}' for '{company_name}' via direct lookup")
        return ticker
        
    print(f"[DEBUG] Could not find ticker for '{company_name}'")
    return ""

def classify_request(state: DispatcherState) -> DispatcherState:
    """
    Classify the user request and extract parameters using LLM
    """
    state_copy = state.copy()
    
    messages = state_copy.get("messages", [])
    if not messages:
        return state_copy
    
    last_message = messages[-1]
    
    if isinstance(last_message, HumanMessage):
        user_input = last_message.content.strip()
        state_copy["query"] = user_input
        
        print(f"\n[DEBUG] Processing query: '{user_input}'")
        
        # Extract financial data using LLM
        extracted_data = extract_financial_data(user_input)
        
        # Set request type
        request_type = extracted_data.get("request_type", "price")
        state_copy["request_type"] = request_type
        print(f"[DEBUG] Request type: {request_type}")
        
        # Get company name or ticker
        company_or_ticker = extracted_data.get("ticker", "")
        print(f"[DEBUG] Extracted company/ticker: '{company_or_ticker}'")
        
        # If we have a company name/ticker, try to get the actual ticker
        ticker = ""
        if company_or_ticker:
            ticker = get_stock_ticker(company_or_ticker)
        
        # Set other parameters
        time_range = extracted_data.get("time_range", "1mo")
        interval = extracted_data.get("interval", "1d")
        
        print(f"[DEBUG] Final ticker: '{ticker}'")
        print(f"[DEBUG] Time range: '{time_range}'")
        print(f"[DEBUG] Interval: '{interval}'")
        
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
    
    # Handle empty ticker case
    if not ticker:
        error_msg = "I couldn't identify a valid stock ticker from your query. Could you please specify the company name more clearly?"
        state_copy["messages"].append(AIMessage(content=error_msg))
        return state_copy
    
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
    
    # Handle empty ticker case
    if not ticker:
        error_msg = "I couldn't identify a valid stock ticker from your query. Could you please specify the company name more clearly?"
        state_copy["messages"].append(AIMessage(content=error_msg))
        return state_copy
    
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

__all__ = ['dispatcher_app', 'DispatcherState'] 