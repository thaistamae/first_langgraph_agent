import os
from dotenv import load_dotenv
import requests
from typing import Dict, Any, List, TypedDict, Optional
from langgraph.graph import Graph, END
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
YAHOO_FINANCE_URL = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/market/v2/get-quotes"
YAHOO_FINANCE_SEARCH_URL = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/auto-complete"

headers = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": "apidojo-yahoo-finance-v1.p.rapidapi.com"
}

# Common company name to ticker mapping
COMPANY_TO_TICKER = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "amazon": "AMZN",
    "tesla": "TSLA",
    "facebook": "META",
    "meta": "META",
    "netflix": "NFLX",
    "nvidia": "NVDA",
}

class AgentState(TypedDict):
    messages: List[HumanMessage | AIMessage]
    current_symbol: str

def search_symbol(query: str) -> Optional[str]:
    """
    Search for a company symbol using Yahoo Finance API
    """
    querystring = {"q": query, "region": "US"}
    response = requests.get(YAHOO_FINANCE_SEARCH_URL, headers=headers, params=querystring)
    data = response.json()
    
    quotes = data.get("quotes", [])
    if quotes and len(quotes) > 0:
        return quotes[0].get("symbol")
    return None

def get_ticker_symbol(query: str) -> str:
    """
    Convert company name to ticker symbol or validate ticker
    """
    if query.isupper() and 1 <= len(query) <= 5:
        return query
    
    normalized_query = query.lower().strip()
    if normalized_query in COMPANY_TO_TICKER:
        return COMPANY_TO_TICKER[normalized_query]
    
    symbol = search_symbol(query)
    if symbol:
        return symbol
    
    return query.upper()

def fetch_company_info(symbol: str) -> Dict[str, Any]:
    """
    Fetch company information from Yahoo Finance API
    """
    querystring = {"symbols": symbol, "region": "US"}
    response = requests.get(YAHOO_FINANCE_URL, headers=headers, params=querystring)
    return response.json()

def process_query(state: AgentState) -> AgentState:
    """
    Process the user query and fetch company information
    """
    messages = state.get("messages", [])
    if not messages:
        return state
        
    last_message = messages[-1]
    
    if isinstance(last_message, HumanMessage):
        user_input = last_message.content.strip()
        
        symbol = get_ticker_symbol(user_input)
        state["current_symbol"] = symbol
        
        try:
            company_data = fetch_company_info(symbol)
            quotes = company_data.get("quoteResponse", {}).get("result", [])
            
            if not quotes:
                state["messages"].append(AIMessage(content=f"Could not find information for '{user_input}'. Please try a different company name or ticker symbol."))
                return state
                
            quote = quotes[0]
            
            response = f"""
            Company Information for {symbol} ({quote.get('longName', 'N/A')}):
            Current Price: ${quote.get('regularMarketPrice', 'N/A')}
            Market Cap: ${quote.get('marketCap', 'N/A')}
            52 Week Range: ${quote.get('fiftyTwoWeekLow', 'N/A')} - ${quote.get('fiftyTwoWeekHigh', 'N/A')}
            P/E Ratio: {quote.get('trailingPE', 'N/A')}
            Dividend Yield: {quote.get('dividendYield', 'N/A')}%
            """
            
            state["messages"].append(AIMessage(content=response))
            
        except Exception as e:
            state["messages"].append(AIMessage(content=f"Error fetching information for '{user_input}': {str(e)}"))

    return state

workflow = Graph()

workflow.add_node("process_query", process_query)

workflow.set_entry_point("process_query")

workflow.add_edge("process_query", END)

app = workflow.compile()

if __name__ == "__main__":
    print("Company Information Agent")
    print("------------------------")
    print("Enter a company name or ticker symbol (e.g., 'Apple' or 'AAPL'):")
    
    user_input = input("> ")
    
    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_input)],
        "current_symbol": ""
    }
        
    try:
        result = app.invoke(initial_state)
        
        if result and "messages" in result:
            for message in result["messages"]:
                if isinstance(message, AIMessage):
                    print(f"\n{message.content}\n")
        else:
            print("Error: No result returned from the workflow")
    except Exception as e:
        print(f"Error running the workflow: {str(e)}")

# Export these for the Gradio interface
__all__ = ['app', 'AgentState', 'get_ticker_symbol', 'search_symbol']