from fastapi import FastAPI
from routers.books import router as router_books
from models.books import BooksModel
from contextlib import asynccontextmanager
from database import engine, Model

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Model.metadata.create_all)
    
    print('Database is ready')

    yield

    print('DataBase done')

app = FastAPI(lifespan=lifespan)
app.include_router(router_books)