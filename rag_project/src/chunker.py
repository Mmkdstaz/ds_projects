from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 512,
    chunk_overlap = 64,
    separators=[ "\n\n", "\n", ". ", " ", "" ]
)

def chunk_text(text: str):
    return splitter.split_text(text)