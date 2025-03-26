import gradio as gr
from langchain_core.messages import HumanMessage
from chart_agent import chart_app, ChartAgentState, fetch_chart_data, create_chart_html
from price_agent import get_ticker_symbol
import re
import pandas as pd
import plotly.graph_objects as go
import datetime

def extract_plotly_data(html_content):
    """
    Extract the data needed for plotly from HTML content
    """
    try:
        data_match = re.search(r'Plotly\.newPlot\(\s*"[^"]+",\s*(\[{.*?\}]\s*),\s*{', html_content, re.DOTALL)
        layout_match = re.search(r'Plotly\.newPlot\(\s*"[^"]+",\s*\[{.*?\}]\s*,\s*({.*?})\s*,', html_content, re.DOTALL)
        
        if data_match and layout_match:
            import json
            data_json = data_match.group(1)
            layout_json = layout_match.group(1)
            
            data_json = re.sub(r'{"dtype":"[^"]+","bdata":"[^"]+"}', '[]', data_json)
            
            data = json.loads(data_json)
            layout = json.loads(layout_json)
            
            return data, layout
    except Exception as e:
        print(f"Error extracting Plotly data: {str(e)}")
    
    return None, None

def direct_chart_creation(ticker, time_range="1mo", interval="1d"):
    """
    Directly create a chart without going through the LangGraph agent
    """
    data = fetch_chart_data(ticker, interval=interval, range=time_range)
    if not data:
        return None, f"Failed to retrieve chart data for {ticker}"
    
    chart_data = data.get("chart", {})
    result = chart_data.get("result", [])
    
    if not result or len(result) == 0:
        return None, f"No chart data available for {ticker}"
    
    timestamps = result[0].get("timestamp", [])
    quotes = result[0].get("indicators", {}).get("quote", [])
    
    if not timestamps or not quotes or len(quotes) == 0:
        return None, f"Incomplete data received for {ticker}"
    
    close_prices = quotes[0].get("close", [])
    
    if not close_prices:
        return None, f"No price data available for {ticker}"
    
    dates = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]
    
    df = pd.DataFrame({
        'Date': dates,
        'Close': close_prices
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name=ticker))
    
    fig.update_layout(
        title=f"{ticker} Stock Price Chart ({time_range})",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white"
    )
    
    return fig, f"Successfully retrieved chart for {ticker}"

def query_chart_agent(company, time_range, interval):
    """
    Query the chart agent with user inputs and return results
    
    Args:
        company: Company name or ticker symbol
        time_range: Time range for chart data
        interval: Interval for data points
        
    Returns:
        HTML for chart or error message
    """
    ticker = get_ticker_symbol(company)
    
    fig, message = direct_chart_creation(ticker, time_range, interval)
    if fig:
        return fig, f"## {message}\n\nTime Range: {time_range}\nInterval: {interval}"
    else:
        return None, message
    
    # NOTE: The code below uses LangGraph but is commented out as direct creation is more reliable for web display
    """
    # Build query with parameters
    user_query = company
    if time_range != "1mo":  # Only add if not default
        user_query += f" range:{time_range}"
    if interval != "1d":  # Only add if not default
        user_query += f" interval:{interval}"
    
    # Initialize agent state
    initial_state: ChartAgentState = {
        "messages": [HumanMessage(content=user_query)],
        "current_symbol": "",
        "chart_data": None,
        "error": None
    }
    
    try:
        # Invoke the agent
        result = chart_app.invoke(initial_state)
        
        if result and "messages" in result and len(result["messages"]) > 1:
            # Get the last AI message
            for message in reversed(result["messages"]):
                if hasattr(message, "type") and message.type == "ai":
                    # Process the response
                    chart_html, text_response = process_agent_response(message.content)
                    
                    if chart_html:
                        # Try to extract plotly data from HTML
                        data, layout = extract_plotly_data(chart_html)
                        if data and layout:
                            fig = go.Figure(data=data, layout=layout)
                            return fig, text_response
                        else:
                            return None, "Could not parse chart data from response."
                    else:
                        return None, text_response
        
        # If no valid response was found
        return None, "The agent did not return a valid response."
    
    except Exception as e:
        return None, f"Error querying the agent: {str(e)}"
    """

def create_interface():
    """
    Create the Gradio interface for the chart agent
    """
    with gr.Blocks(title="Stock Chart Agent") as interface:
        gr.Markdown("# Historical Stock Chart Agent")
        gr.Markdown("Enter a company name or ticker symbol to view historical price charts")
        
        with gr.Row():
            with gr.Column(scale=1):
                company_input = gr.Textbox(
                    label="Company Name or Ticker Symbol",
                    placeholder="e.g., Apple or AAPL",
                    interactive=True
                )
                
                with gr.Row():
                    time_range = gr.Dropdown(
                        label="Time Range",
                        choices=["1d", "5d", "1mo", "3mo", "6mo", "1y", "5y", "max"],
                        value="1mo",
                        interactive=True
                    )
                    
                    interval = gr.Dropdown(
                        label="Interval",
                        choices=["1d", "1wk", "1mo"],
                        value="1d",
                        interactive=True
                    )
                
                submit_btn = gr.Button("Get Chart")
            
        with gr.Row():
            with gr.Column():
                chart_output = gr.Plot(label="Stock Price Chart")
                text_output = gr.Markdown(label="Information")
        
        submit_btn.click(
            fn=query_chart_agent,
            inputs=[company_input, time_range, interval],
            outputs=[chart_output, text_output]
        )
        
        gr.Examples(
            examples=[
                ["Apple", "3mo", "1d"],
                ["MSFT", "6mo", "1wk"],
                ["Tesla", "1y", "1d"],
                ["AMZN", "1mo", "1d"],
                ["NVDA", "5d", "1d"]
            ],
            inputs=[company_input, time_range, interval]
        )
        
        interface.theme = gr.themes.Soft()
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch() 