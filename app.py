import os
import gradio as gr
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from main import app, AgentState, get_ticker_symbol

load_dotenv()

def convert_to_chat_format(company_info: str) -> List[Tuple[str, str]]:
    """Convert agent response to chat format for Gradio"""
    lines = company_info.strip().split('\n')
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
        formatted_html += f"<p>{company_info}</p>"
    
    formatted_html += "</div>"
    
    return [(None, formatted_html)]

def get_company_info(company_input: str) -> List[Tuple[str, str]]:
    """Get company information and return in chat format"""
    if not company_input:
        return [(None, "<p>Please enter a company name or ticker symbol.</p>")]
    
    ticker = get_ticker_symbol(company_input)
    
    initial_state: AgentState = {
        "messages": [HumanMessage(content=company_input)],
        "current_symbol": ""
    }
    
    try:
        result = app.invoke(initial_state)
        
        if result and "messages" in result:
            for message in result["messages"]:
                if isinstance(message, AIMessage):
                    return convert_to_chat_format(message.content)
        
        return [(None, "<p>Error: No information found for this company.</p>")]
    
    except Exception as e:
        return [(None, f"<p>Error: {str(e)}</p>")]

with gr.Blocks(theme=gr.themes.Soft(), title="Company Info Agent") as demo:
    gr.Markdown("""
    # 🏢 Company Information Agent
    
    Enter a company name (e.g., "Apple") or ticker symbol (e.g., "AAPL") to get the latest financial information.
    """)
    
    with gr.Row():
        with gr.Column():
            company_input = gr.Textbox(
                label="Company Name or Ticker Symbol",
                placeholder="Enter company name or ticker (e.g., Apple, MSFT, Tesla)",
                lines=1
            )
            submit_btn = gr.Button("Get Info", variant="primary")
        
    chatbot = gr.Chatbot(
        label="Company Information",
        height=400,
        show_label=True,
        show_share_button=False,
        avatar_images=(None, None)
    )
    
    # Set up event handlers
    submit_btn.click(
        fn=get_company_info,
        inputs=company_input,
        outputs=chatbot
    )
    company_input.submit(
        fn=get_company_info,
        inputs=company_input,
        outputs=chatbot
    )
    
    gr.Markdown("""
    ### How to use
    1. Type a company name or ticker symbol in the input box
    2. Press "Get Info" or hit Enter
    3. View the latest financial information for the company
    
    ### Examples
    - Apple
    - MSFT
    - Tesla
    - AMZN
    - Google
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(share=False) 