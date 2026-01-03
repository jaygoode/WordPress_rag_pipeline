import html
import re

def clean_text(text: str) -> str:
    if not text:
        return ""

    # Decode escaped HTML codes
    text = html.unescape(text)
    text = text.replace("\x00", "")

    # fix line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # excessive indentation
    text = re.sub(r"\n[ \t]+", "\n", text)

    # blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalize spaces (but NOT newlines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()
