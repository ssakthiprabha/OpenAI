from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from openai import OpenAI
import json
import requests
import re
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
client = OpenAI(api_key="Replace with your API key")  # Replace with your API key

class AgenticAI:
    def __init__(self):
        self.conversation_history = []
        self.tools = {
            "get_weather": self.get_weather,
            "calculate": self.calculate,
            "search_web": self.search_web,
            "get_current_time": self.get_current_time,
            "create_reminder": self.create_reminder
        }
        
        self.system_prompt = """
        You are an intelligent agentic AI assistant that can use various tools to help users.
        
        Available tools:
        1. get_weather(city) - Get current weather for a city
        2. calculate(expression) - Perform mathematical calculations
        3. search_web(query) - Search the web for information
        4. get_current_time() - Get current date and time
        5. create_reminder(text, time) - Create a reminder
        
        When you need to use a tool, format your response as:
        TOOL_CALL: tool_name(parameters)
        
        Always explain your reasoning and what tool you're using before making the call.
        After getting results, integrate them naturally into your response.
        
        Be conversational and helpful while demonstrating autonomous decision-making.
        """
    
    def get_weather(self, city):
        """Simulate weather API call (replace with real API in production)"""
        # This is a mock function - in real implementation, use actual weather API
        weather_data = {
            "london": {"temp": "15°C", "condition": "Cloudy", "humidity": "78%"},
            "new york": {"temp": "22°C", "condition": "Sunny", "humidity": "65%"},
            "tokyo": {"temp": "18°C", "condition": "Rainy", "humidity": "85%"},
            "default": {"temp": "20°C", "condition": "Partly cloudy", "humidity": "70%"}
        }
        
        city_key = city.lower() if city.lower() in weather_data else "default"
        return weather_data[city_key]
    
    def calculate(self, expression):
        """Safely evaluate mathematical expressions"""
        try:
            # Remove any non-mathematical characters for safety
            safe_expr = re.sub(r'[^0-9+\-*/().\s]', '', expression)
            result = eval(safe_expr)
            return f"Result: {result}"
        except Exception as e:
            return f"Error in calculation: {str(e)}"
    
    def search_web(self, query):
        """Simulate web search (replace with real search API in production)"""
        # Mock search results
        mock_results = {
            "python": "Python is a high-level programming language known for its simplicity and readability.",
            "ai": "Artificial Intelligence refers to machine intelligence that can perform tasks typically requiring human intelligence.",
            "default": f"Search results for '{query}': This is a simulated search result. In production, integrate with a real search API."
        }
        
        for key in mock_results:
            if key in query.lower():
                return mock_results[key]
        return mock_results["default"]
    
    def get_current_time(self):
        """Get current date and time"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def create_reminder(self, text, time):
        """Create a reminder (simulate saving to database)"""
        return f"Reminder created: '{text}' for {time}"
    
    def parse_tool_calls(self, text):
        """Parse tool calls from AI response"""
        tool_pattern = r'TOOL_CALL:\s*(\w+)\(([^)]*)\)'
        matches = re.findall(tool_pattern, text)
        return matches
    
    def execute_tool(self, tool_name, params):
        """Execute a tool with given parameters"""
        if tool_name not in self.tools:
            return f"Unknown tool: {tool_name}"
        
        try:
            # Parse parameters (simple implementation)
            if params:
                # Handle string parameters
                if params.startswith('"') and params.endswith('"'):
                    params = params[1:-1]
                elif params.startswith("'") and params.endswith("'"):
                    params = params[1:-1]
            
            result = self.tools[tool_name](params) if params else self.tools[tool_name]()
            return result
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"
    
    def process_message(self, user_message):
        """Process user message and generate response with tool usage"""
        
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Create OpenAI API call
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    *self.conversation_history[-10:]  # Keep last 10 messages for context
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            
            # Check for tool calls in the response
            tool_calls = self.parse_tool_calls(ai_response)
            
            # Execute tools and integrate results
            final_response = ai_response
            tool_results = []
            
            for tool_name, params in tool_calls:
                result = self.execute_tool(tool_name, params)
                tool_results.append({
                    "tool": tool_name,
                    "params": params,
                    "result": result
                })
                
                # Replace tool call with result in the response
                tool_call_pattern = f"TOOL_CALL: {tool_name}\\({re.escape(params)}\\)"
                final_response = re.sub(tool_call_pattern, f"[Tool Result: {result}]", final_response)
            
            # Add AI response to history
            self.conversation_history.append({"role": "assistant", "content": final_response})
            
            return {
                "response": final_response,
                "tool_calls": tool_results,
                "reasoning": "AI analyzed the request and autonomously decided which tools to use."
            }
            
        except Exception as e:
            return {
                "response": f"I apologize, but I encountered an error: {str(e)}. Please make sure your OpenAI API key is set correctly.",
                "tool_calls": [],
                "reasoning": "Error occurred during processing."
            }

# Initialize the agent
agent = AgenticAI()

@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Agentic AI Demo</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .chat-box { height: 400px; border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; overflow-y: scroll; background: #fafafa; border-radius: 5px; }
            .message { margin-bottom: 15px; padding: 10px; border-radius: 5px; }
            .user { background: #e3f2fd; border-left: 4px solid #2196f3; }
            .assistant { background: #f3e5f5; border-left: 4px solid #9c27b0; }
            .tool-call { background: #fff3e0; border-left: 4px solid #ff9800; margin: 5px 0; padding: 8px; font-size: 0.9em; }
            input[type="text"] { width: 70%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            button { width: 25%; padding: 10px; background: #2196f3; color: white; border: none; border-radius: 5px; cursor: pointer; margin-left: 2%; }
            button:hover { background: #1976d2; }
            .examples { margin-top: 20px; }
            .example { background: #e8f5e8; padding: 10px; margin: 5px 0; border-radius: 5px; cursor: pointer; }
            .example:hover { background: #d4edda; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Agentic AI Proof of Concept</h1>
            <p>This AI agent can autonomously use tools to help you. Try asking for weather, calculations, web searches, or the current time!</p>
            
            <div class="chat-box" id="chatBox"></div>
            
            <div>
                <input type="text" id="messageInput" placeholder="Ask me anything..." onkeypress="if(event.key==='Enter') sendMessage()">
                <button onclick="sendMessage()">Send</button>
            </div>
            
            <div class="examples">
                <h3>Try these examples:</h3>
                <div class="example" onclick="setMessage('What\\'s the weather like in London?')">
                    What's the weather like in London?
                </div>
                <div class="example" onclick="setMessage('Calculate 15% tip on a $87 bill')">
                    Calculate 15% tip on a $87 bill
                </div>
                <div class="example" onclick="setMessage('What time is it now?')">
                    What time is it now?
                </div>
                <div class="example" onclick="setMessage('Search for information about Python programming')">
                    Search for information about Python programming
                </div>
            </div>
        </div>

        <script>
            function addMessage(sender, content, toolCalls = []) {
                const chatBox = document.getElementById('chatBox');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}`;
                
                let html = `<strong>${sender === 'user' ? 'You' : 'AI Agent'}:</strong> ${content}`;
                
                if (toolCalls.length > 0) {
                    html += '<div style="margin-top: 10px;"><strong>Tools Used:</strong></div>';
                    toolCalls.forEach(tool => {
                        html += `<div class="tool-call">🔧 ${tool.tool}(${tool.params}) → ${tool.result}</div>`;
                    });
                }
                
                messageDiv.innerHTML = html;
                chatBox.appendChild(messageDiv);
                chatBox.scrollTop = chatBox.scrollHeight;
            }

            function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                
                if (!message) return;
                
                addMessage('user', message);
                input.value = '';
                
                // Show thinking indicator
                addMessage('assistant', '🤔 Thinking and selecting appropriate tools...');
                
                fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                })
                .then(response => response.json())
                .then(data => {
                    // Remove thinking indicator
                    const chatBox = document.getElementById('chatBox');
                    chatBox.removeChild(chatBox.lastChild);
                    
                    addMessage('assistant', data.response, data.tool_calls);
                })
                .catch(error => {
                    console.error('Error:', error);
                    addMessage('assistant', 'Sorry, I encountered an error. Please try again.');
                });
            }

            function setMessage(message) {
                document.getElementById('messageInput').value = message;
            }

            // Add welcome message
            addMessage('assistant', 'Hello! I\\'m an agentic AI that can autonomously use various tools to help you. I can check weather, perform calculations, search for information, tell you the time, and more. What would you like me to help you with?');
        </script>
    </body>
    </html>
    """)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    response = agent.process_message(user_message)
    return jsonify(response)

@app.route('/history')
def get_history():
    return jsonify({"history": agent.conversation_history})

@app.route('/clear')
def clear_history():
    agent.conversation_history = []
    return jsonify({"status": "History cleared"})

if __name__ == '__main__':
    print("🤖 Starting Agentic AI Server...")
    print("📝 Don't forget to set your OpenAI API key in the code!")
    print("🌐 Visit http://localhost:5000 to interact with the agent")
    app.run(debug=True, port=5000)