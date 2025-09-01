# Specialized Transcription Summary Agent ("Suhail Summary Agent")
# This agent runs automatically after a call's transcript is processed
# Purpose: Convert raw transcript into structured summary with actionable insights

from langchain_openai import ChatOpenAI
import os
from datetime import datetime
import re


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


def generate_meeting_summary(transcript_text, meeting_metadata=None):
    """
    Generate a comprehensive meeting summary from audio transcript using Suhail Summary Agent format.
    
    Args:
        transcript_text: str - The raw transcript text from audio transcription
        meeting_metadata: dict - Contains meeting info like title, duration, participants, etc.
        
    Returns:
        str: Structured summary in the specified format
    """
    
    # Extract metadata with defaults
    if meeting_metadata is None:
        meeting_metadata = {}
        
    customer_name = meeting_metadata.get('customer_name', '[Customer Name Not Specified]')
    title = meeting_metadata.get('title', 'Live Meeting')
    duration = meeting_metadata.get('duration', 'Unknown')
    participants = meeting_metadata.get('participants', 'Unknown participants')
    date_time = meeting_metadata.get('date_time', datetime.now().strftime('%d/%m/%Y, %H:%M'))
    recording_link = meeting_metadata.get('recording_link', '[Recording Available]')
    
    prompt = f"""You are Suhail, an expert sales conversation analyst. Analyze this meeting transcript and provide a comprehensive summary following this exact structure:

**MEETING TRANSCRIPT SUMMARY**

**1. Call Metadata**
- Customer Name: {customer_name}
- Meeting Title: {title}
- Attendees: {participants}
- Date & Time: {date_time}
- Duration: {duration}
- Recording Link: {recording_link}
- Transcript: [Expand to view full transcript]

**2. Key Points Discussed**
[Extract 3-5 main topics discussed in the meeting, format as:]
• **Topic 1** → [Concise summary of customer input + seller response]
• **Topic 2** → [Summary]
• **Topic 3** → [Summary]
(Continue as needed)

**3. Actions from the Call**
[Extract 3-5 actionable items from the meeting, format as:]
• **Action 1**: [Description] → [Suggested workflow action]
  Example: Update CRM with deal status: "Proposal Stage"
• **Action 2**: [Description] → [Suggested workflow action]
  Example: Create new offer for Customer X
• **Action 3**: [Description] → [Suggested workflow action]
  Example: Schedule follow-up call on [specific date]

**4. Suhail's Analysis of the Meeting**

**4.1 What Went Right**
→ [Positive highlights, successful moments, customer engagement points]

**4.2 What Went Wrong**
→ [Risks, gaps, unresolved issues, missed opportunities]

**4.3 Sentiment Analysis**
- **Overall Sentiment**: [Positive / Neutral / Negative]
- **Confidence Level**: [High / Medium / Low]
- **Key Sentiment Drivers**: [What drove the sentiment - satisfaction, concerns, etc.]

**4.4 Key Learning Points**
- **Best Practices**: [What worked well in this conversation]
- **Areas for Improvement**: [What could be done better next time]

**5. Next Steps & Recommendations (By Suhail)**
• [Specific recommendation 1 with timeline]
• [Specific recommendation 2 with timeline]
• [Specific recommendation 3 with timeline]

---

**TRANSCRIPT TO ANALYZE:**
{transcript_text}

Please provide a comprehensive analysis following the exact structure above. Make the summary actionable and insightful for sales professionals."""

    llm = ChatOpenAI(model='gpt-4o', temperature=0.2, api_key=os.getenv('OPENAI_API_KEY'))
    
    response = llm.invoke(prompt)
    return response.content if hasattr(response, 'content') else str(response)


def generate_summary(transcript):
    """
    Generate a summary from a conversation transcript using an LLM.
    This is the original function for backward compatibility.
    
    Args:
        transcript: str - The formatted transcript to summarize
        
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
