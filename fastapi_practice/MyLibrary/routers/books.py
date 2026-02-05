from fastapi import APIRouter, status
from typing import Annotated
from schemas.books import SBookAdd, SBook
from database import SessionDep
from repository import BooksRepository

router = APIRouter(prefix='/books', tags=['Книги'])

@router.post('', response_model=SBook, status_code=status.HTTP_201_CREATED)
async def add_book(book: SBookAdd, session: SessionDep):
    book_model = await BooksRepository.add_book(book, session)
    return book_model

@router.get('', status_code=status.HTTP_200_OK)
async def get_all_books(session: SessionDep):
    book_model = await BooksRepository.get_all(session)
    return book_model

@router.get('/{id}', response_model = SBook)
async def get_book(id: int, session: SessionDep):
    book_model = await BooksRepository.get_book(id, session)
    return book_model

@router.delete('/{id}', status_code=status.HTTP_204_NO_CONTENT)
async def del_book(id: int, session: SessionDep):
    book_model = await BooksRepository.del_book(id, session)
    return book_model

@router.put('/{id}', response_model = SBook)
async def update_book(id: int,book: SBookAdd ,session: SessionDep):
    book_model = await BooksRepository.update_book(id, book, session)
    return book_model