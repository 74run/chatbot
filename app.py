from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chatbot API")

class ChatRequest(BaseModel):
    user_input: str

class ChatResponse(BaseModel):
    response: str
    conversation_id: str

class Conversation:
    def __init__(self):
        self.history: List[Dict[str, str]] = []
        self.max_history = 5
        
    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history * 2:  # Count both user and assistant messages
            self.history = self.history[-self.max_history * 2:]
            
    def get_formatted_history(self) -> str:
        formatted = []
        for msg in self.history:
            prefix = "Human: " if msg["role"] == "user" else "Assistant: "
            formatted.append(f"{prefix}{msg['content']}")
        return "\n".join(formatted)

class ChatbotManager:
    def __init__(self):
        self.conversations: Dict[str, Conversation] = {}
        try:
            # Using a better model for chat
            model_name = "facebook/blenderbot-400M-distill"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = pipeline(
                "text2text-generation",
                model=model_name,
                tokenizer=self.tokenizer,
                device="cpu"  # Change to "cuda" if using GPU
            )
            logger.info("Chat model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load chat model: {e}")
            raise RuntimeError("Failed to initialize chatbot")

    def generate_response(self, conversation: Conversation, user_input: str) -> str:
        try:
            # Format the conversation history and current input
            context = conversation.get_formatted_history()
            
            # Generate response
            response = self.model(
                user_input,
                max_length=100,
                num_return_sequences=1,
                clean_up_tokenization_spaces=True
            )[0]["generated_text"]
            
            # Clean up the response
            response = response.strip()
            
            # Fallback for empty or very short responses
            if len(response) < 2:
                response = "I apologize, I'm not sure how to respond to that. Could you please rephrase your question?"
            
            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate response")

chatbot = ChatbotManager()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Validate input
        user_input = request.user_input.strip()
        if not user_input:
            raise HTTPException(status_code=400, detail="Empty user input")
            
        # Create or get conversation
        conversation_id = "default"  # In production, generate unique IDs per user
        if conversation_id not in chatbot.conversations:
            chatbot.conversations[conversation_id] = Conversation()
        
        conversation = chatbot.conversations[conversation_id]
        
        # Add user message to history
        conversation.add_message("user", user_input)
        
        # Generate response
        ai_response = chatbot.generate_response(conversation, user_input)
        
        # Add AI response to history
        conversation.add_message("assistant", ai_response)
        
        return ChatResponse(
            response=ai_response,
            conversation_id=conversation_id
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Add endpoint to clear conversation history
@app.post("/clear_conversation/{conversation_id}")
async def clear_conversation(conversation_id: str):
    if conversation_id in chatbot.conversations:
        chatbot.conversations[conversation_id] = Conversation()
        return {"status": "conversation cleared"}
    raise HTTPException(status_code=404, detail="Conversation not found")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "facebook/blenderbot-400M-distill"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)