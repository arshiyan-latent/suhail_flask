# Summary Agent:

#first function: --> extract transcript from chat messages
#second function: --> generate summary from transcript
from langchain_openai import ChatOpenAI
import os


def extract_transcript(chat_messages):
    """
    Extract a formatted transcript from chat messages.
    
    Args:
        chat_messages: List of ChatMessage objects from the database
        
    Returns:
        str: Formatted transcript with speaker labels and timestamps
    """
    transcript = ""
    
    for message in chat_messages:
        # Format: [Timestamp] Speaker: Message
        timestamp = message.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        speaker = "User" if message.sender == "user" else "Bot"
        transcript += f"[{timestamp}] {speaker}: {message.message}\n"
    
    return transcript.strip()


def generate_summary(transcript):
    """
    Generate a summary from a conversation transcript using an LLM.
    
    Args:
        transcript: str - The formatted transcript to summarize
        llm: Language model to use for summarization
        
    Returns:
        str: Summary of the conversation
    """
    prompt = f"""You are a helpful assistant that summarizes conversations. 
    
Please provide a concise summary of the following conversation transcript. 
Make sure to include:
- Client name if available
- Key topics discussed
- Action items or next steps
- Important decisions made

Transcript:
{transcript}

Summary:"""
    llm = ChatOpenAI(model='gpt-4o', temperature=0.2, api_key=os.getenv('OPENAI_API_KEY'))
    
    response = llm.invoke(prompt)
    return response.content if hasattr(response, 'content') else str(response)
