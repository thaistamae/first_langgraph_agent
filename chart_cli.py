from langchain_core.messages import HumanMessage
from chart_agent import chart_app, ChartAgentState

def main():
    print("Historical Stock Chart Agent")
    print("---------------------------")
    print("Enter a company name or ticker symbol (e.g., 'Apple' or 'AAPL')")
    print("You can also specify time range and interval:")
    print("  e.g., 'Apple range:3mo interval:1d'")
    print("Available ranges: 1d, 5d, 1mo, 3mo, 6mo, 1y, 5y, max")
    print("Available intervals: 1d, 1wk, 1mo")
    print("Type 'quit' to exit")
    print()
    
    while True:
        user_input = input("> ")
        
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        
        if not user_input.strip():
            continue
        
        initial_state: ChartAgentState = {
            "messages": [HumanMessage(content=user_input)],
            "current_symbol": "",
            "chart_data": None,
            "error": None
        }
        
        try:
            result = chart_app.invoke(initial_state)
            
            if result and "messages" in result:
                for message in result["messages"]:
                    if hasattr(message, "content") and hasattr(message, "type") and message.type == "ai":
                        print(f"\n{message.content}\n")
            else:
                print("Error: No result returned from the agent")
        except Exception as e:
            print(f"Error running the agent: {str(e)}")

if __name__ == "__main__":
    main() 