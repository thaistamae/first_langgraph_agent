import os
import gradio as gr
import re
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple, Optional
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from dispatcher_agent import dispatcher_app, DispatcherState, detect_request_type
from chart_web import extract_plotly_data, direct_chart_creation

load_dotenv()

def convert_to_chat_format(content: str) -> List[Tuple[str, str]]:
    """Convert agent response to chat format for Gradio"""
    # Check if this is a chart response (contains HTML for plotly)
    if "Stock Chart for" in content and "<div id=" in content:
        # Extract the text portion before the chart
        text_part = content.split("<div id=")[0].strip()
        
        # Create HTML for the text part
        formatted_html = f"<div style='background-color: #f0f8ff; padding: 15px; border-radius: 10px;'>{text_part}</div>"
        
        return [(None, formatted_html)]
    
    # Handle price information response
    lines = content.strip().split('\n')
    formatted_html = "<div style='background-color: #f0f8ff; padding: 15px; border-radius: 10px;'>"
    
    if lines and "Company Information" in lines[0]:
        title_parts = lines[0].split("(")
        symbol = title_parts[0].replace("Company Information for", "").strip()
        company_name = title_parts[1].replace(")", "").strip() if len(title_parts) > 1 else symbol
        
        formatted_html += f"<h3>📊 {company_name} ({symbol})</h3>"
        
        formatted_html += "<table style='width:100%; border-collapse: collapse;'>"
        
        for line in lines[1:]:
            if ":" in line:
                key, value = line.split(":", 1)
                formatted_html += f"""
                <tr>
                    <td style='padding: 8px; border-bottom: 1px solid #ddd; font-weight: bold;'>{key.strip()}</td>
                    <td style='padding: 8px; border-bottom: 1px solid #ddd;'>{value.strip()}</td>
                </tr>
                """
        
        formatted_html += "</table>"
    else:
        formatted_html += f"<p>{content}</p>"
    
    formatted_html += "</div>"
    
    return [(None, formatted_html)]

def extract_chart_from_response(content: str) -> Optional[go.Figure]:
    """Extract a plotly chart from the response if available"""
    if "<div id=" not in content:
        return None
    
    # Extract the plotly part
    html_content = content.split("<div id=")[1]
    html_content = "<div id=" + html_content
    
    # Try to extract plotly data from HTML
    data, layout = extract_plotly_data(html_content)
    if data and layout:
        fig = go.Figure(data=data, layout=layout)
        return fig
    
    return None

def process_user_query(query: str) -> Tuple[List[Tuple[str, str]], Optional[go.Figure]]:
    """Process user query and return chat format and chart (if available)"""
    if not query:
        return [(None, "<p>Please enter a company name or query.</p>")], None
    
    # Initialize agent state
    initial_state: DispatcherState = {
        "messages": [HumanMessage(content=query)],
        "request_type": None,
        "query": "",
        "ticker_symbol": "",
        "time_range": None,
        "interval": None,
        "price_result": None,
        "chart_result": None
    }
    
    try:
        # Invoke the dispatcher agent
        result = dispatcher_app.invoke(initial_state)
        
        # Get the AI response and check if it's a chart request
        ai_content = None
        is_chart_request = False
        ticker = ""
        time_range = "1mo"
        interval = "1d"
        
        if result:
            # Extract parameters from result
            if "request_type" in result and result["request_type"] == "chart":
                is_chart_request = True
            
            if "ticker_symbol" in result:
                ticker = result["ticker_symbol"]
                
            if "time_range" in result and result["time_range"]:
                time_range = result["time_range"]
                
            if "interval" in result and result["interval"]:
                interval = result["interval"]
                
            # Get AI response
            if "messages" in result and len(result["messages"]) > 1:
                for message in result["messages"]:
                    if isinstance(message, AIMessage):
                        ai_content = message.content
        
        if not ai_content:
            return [(None, "<p>No response from the agent.</p>")], None
        
        # For chart requests, directly create a chart using chart_web's function
        if is_chart_request and ticker:
            fig, message = direct_chart_creation(ticker, time_range, interval)
            if fig:
                # Create a formatted message for the chatbot
                formatted_message = f"""
                ## Stock Chart for {ticker}
                
                Time Range: {time_range}
                Interval: {interval}
                """
                chat_response = [(None, f"<div style='background-color: #f0f8ff; padding: 15px; border-radius: 10px;'>{formatted_message}</div>")]
                return chat_response, fig
        
        # If direct chart creation failed or this is a price request, use the agent response
        chart = extract_chart_from_response(ai_content)
        chat_response = convert_to_chat_format(ai_content)
        
        return chat_response, chart
    
    except Exception as e:
        return [(None, f"<p>Error: {str(e)}</p>")], None

def create_interface():
    """
    Create the unified Gradio interface
    """
    with gr.Blocks(theme=gr.themes.Soft(), title="Financial Data Agent") as demo:
        gr.Markdown("""
        # 📈 Financial Data Agent
        
        Ask for stock price information or historical charts for any company.
        
        Examples:
        - "What's the current price of Apple?"
        - "Show me a chart for Tesla over the last 6 months"
        - "Microsoft stock price"
        - "Amazon historical performance"
        """)
        
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(
                    label="Your Query",
                    placeholder="Enter a company name or ask about a stock (e.g., 'Apple stock price' or 'Show me a chart for Tesla')",
                    lines=2
                )
                submit_btn = gr.Button("Get Information", variant="primary")
        
        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="Information",
                    height=300,
                    show_label=True,
                    show_share_button=False,
                    avatar_images=(None, None)
                )
            with gr.Column(scale=1):
                chart_output = gr.Plot(label="Stock Chart", visible=True)
        
        def handle_query(query):
            chat_response, chart = process_user_query(query)
            
            # Show the appropriate outputs
            if chart:
                return chat_response, chart
            else:
                # Return empty chart if no chart data
                return chat_response, None
        
        # Set up event handlers
        submit_btn.click(
            fn=handle_query,
            inputs=query_input,
            outputs=[chatbot, chart_output]
        )
        query_input.submit(
            fn=handle_query,
            inputs=query_input,
            outputs=[chatbot, chart_output]
        )
        
        gr.Markdown("""
        ### How to use
        1. Type your query about a company or stock in the input box
        2. Press "Get Information" or hit Enter
        3. View the information or chart
        
        ### Example queries
        - "What's the current price of Apple?"
        - "Show me a chart for Tesla"
        - "MSFT stock price"
        - "Amazon chart over the last 3 months"
        - "NVDA historical performance"
        """)
    
    return demo

# Launch the app
if __name__ == "__main__":
    interface = create_interface()
    interface.launch() 