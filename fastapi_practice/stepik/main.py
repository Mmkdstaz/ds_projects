from fastapi import FastAPI
from routers.task import router as tasks_router
from models.task import TasksModel
from contextlib import asynccontextmanager
from database import engine, Model
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Model.metadata.create_all)
    
    print("База данных готова к работе")
    
    yield  
    
    print("Выключение сервера")

app = FastAPI(lifespan=lifespan)
app.include_router(tasks_router)

