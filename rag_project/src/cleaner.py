import re

def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?m)^\d{1,3}$", "", text)   # только одиночные номера строк
    text = re.sub(r"[■□▪▫]", "", text)
    text = re.sub(r"(?im)^.*copyright.*$", "", text)
    return text.strip()