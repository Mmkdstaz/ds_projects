import re

def clean_text(text: str) -> str:

    text = re.sub(r"\n{3,}", "\n\n", text)

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"\n\d+\n", "\n", text)

    text = re.sub(r"[■□▪▫]", "", text)

    text = re.sub(r"(?im)^.*copyright.*$", "", text)
    
    return text.strip()