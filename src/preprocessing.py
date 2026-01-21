import re
from bs4 import BeautifulSoup


def preprocess_text(text):
    """
    Preprocessing rules:
    - Remove HTML & headers
    - Normalize whitespace
    - Lowercase text
    - Replace URLs and emails with tokens
    - Keep punctuation signals (! ? $)
    """

    if not isinstance(text, str):
        return ""

    # 1. Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # 2. Remove common email headers
    text = re.sub(r'(subject|from|to|date):', ' ', text, flags=re.IGNORECASE)

    # 3. Replace URLs with token
    text = re.sub(r'(http\S+|www\S+)', ' URL ', text)

    # 4. Replace email addresses with token
    text = re.sub(r'\S+@\S+', ' EMAIL ', text)

    # 5. Lowercase
    text = text.lower()

    # 6. Keep punctuation signals, remove other junk
    text = re.sub(r'[^a-z0-9!?$]', ' ', text)

    # 7. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text
