import re

def sanitize_text(text):
    """Sanitize text to ASCII, removing or replacing problematic characters including emojis"""
    if not isinstance(text, str):
        text = str(text)
        
    # Remove emojis and other special characters
    emoji_pattern = re.compile("[" 
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Convert to ASCII, replacing non-ASCII characters
    return text.encode('ascii', errors='replace').decode('ascii')