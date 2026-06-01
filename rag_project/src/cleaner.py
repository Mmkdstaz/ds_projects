import re


def clean_text(text: str) -> str:
    text = text.replace("\u201a", ",")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')

    ocr_fixes = {
        r"\bmo\b": "по", r"\bMO\b": "по",
        r"\bMX\b": "их", r"\bOHH\b": "они",
        r"\bOHM\b": "они", r"\bOHA\b": "она",
        r"\bHa\b": "на", r"\bHO\b": "по",
        r"\bHe\b": "не", r"\bWIA\b": "для",
        r"\bTOM\b": "том", r"\bTEМ\b": "тем",
        r"\bOT\b": "от", r"\bBO\b": "во",
        r"\bCO\b": "со", r"\bOBI\b": "бы",
        r"\bTak\b": "так", r"\bKak\b": "как",
        r"\bIlo\b": "По", r"\bMae\b": "мае",
        r"\bnog\b": "под", r"\bJia\b": "для",
        r"\bMHe\b": "мне",
    }
    for pattern, replacement in ocr_fixes.items():
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"(?<!\w)\|(?!\w)", "", text)
    text = re.sub(r"-\n\s*", "", text)
    text = re.sub(r"[■□▪▫@#\\]", "", text)
    text = re.sub(r"(?m)^\d{1,3}$", "", text)
    text = re.sub(r"(?im)^.*copyright.*$", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()