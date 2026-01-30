from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from nnTag import FastTagPredictor
import os

app = FastAPI(title="Tag Predictor API")

predictor = FastTagPredictor()
MODEL_PATH = 'pytorch_tag_model'

if os.path.exists(f"{MODEL_PATH}.pth"):
    predictor.load(MODEL_PATH)
else:
    print("ВНИМАНИЕ: Файл модели не найден!")

class TextRequest(BaseModel):
    text: str
    threshold: float = 0.1

class TagResponse(BaseModel):
    name: str
    probability: float

@app.post("/predict")
async def predict_tags(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")
    
    tags = predictor.predict(
        request.text, 
        threshold=request.threshold,
        boost_factor = 2
    )
    return [t[0] for t in tags][:10]


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.get("/", response_class=HTMLResponse)
async def read_index():
    file_path = os.path.join(BASE_DIR, "templates/index.html")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Файл {file_path} не найден")
        
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)