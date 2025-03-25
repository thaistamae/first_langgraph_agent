import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from chart_agent import fetch_chart_data
from main import get_ticker_symbol

def process_chart_request(company_input, time_range="1mo", interval="1d"):
    """
    Process the chart request from the Gradio interface
    
    Args:
        company_input: Company name or ticker symbol
        time_range: Time range for data (e.g., 1d, 5d, 1mo, 3mo, 6mo, 1y, 5y, max)
        interval: Data interval (e.g., 1d, 1wk, 1mo)
        
    Returns:
        Plotly figure object or error message
    """
    ticker = get_ticker_symbol(company_input)
    
    data = fetch_chart_data(ticker, interval=interval, range=time_range)
    
    if not data:
        return f"Failed to retrieve chart data for {company_input} ({ticker})"
    
    chart_data = data.get("chart", {})
    result = chart_data.get("result", [])
    
    if not result or len(result) == 0:
        return f"No chart data available for {ticker}"
    
    timestamps = result[0].get("timestamp", [])
    quotes = result[0].get("indicators", {}).get("quote", [])
    
    if not timestamps or not quotes or len(quotes) == 0:
        return f"Incomplete data received for {ticker}"
    
    close_prices = quotes[0].get("close", [])
    
    if not close_prices:
        return f"No price data available for {ticker}"
    
    import datetime
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
    
    return fig

def create_interface():
    with gr.Blocks(title="Stock Chart Viewer") as interface:
        gr.Markdown("# Stock Chart Viewer")
        gr.Markdown("Enter a company name or ticker symbol to view its stock price chart")
        
        with gr.Row():
            with gr.Column():
                company_input = gr.Textbox(label="Company Name or Ticker Symbol", placeholder="e.g., Apple or AAPL")
                
                with gr.Row():
                    time_range = gr.Dropdown(
                        label="Time Range",
                        choices=["1d", "5d", "1mo", "3mo", "6mo", "1y", "5y", "max"],
                        value="1mo"
                    )
                    interval = gr.Dropdown(
                        label="Interval",
                        choices=["1d", "1wk", "1mo"],
                        value="1d"
                    )
                
                submit_btn = gr.Button("Get Chart")
        
        output = gr.Plot(label="Chart Output")
        
        submit_btn.click(
            fn=process_chart_request,
            inputs=[company_input, time_range, interval],
            outputs=output
        )
        
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True) 