from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException, status
from models.books import BooksModel
from schemas.books import SBookAdd

class BooksRepository:
    @classmethod
    async def add_book(cls, data: SBookAdd, session: AsyncSession):
        book_dict = data.model_dump()

        book = BooksModel(**book_dict)

        session.add(book)

        await session.commit()
        await session.refresh(book)

        return book
    
    @classmethod
    async def get_all(cls, session: AsyncSession):
        query = select(BooksModel)
        result = await session.execute(query)
        return result.scalars().all()
    
    @classmethod
    async def get_book(cls, id: int, session: AsyncSession):
        query = select(BooksModel).where(BooksModel.id == id)

        result = await session.execute(query)

        book =  result.scalar_one_or_none()

        if book is None:
            raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Book not found"
            )
        
        return book
    
    @classmethod
    async def update_book(cls, id: int, book: SBookAdd, session: AsyncSession):
        query = (
            update(BooksModel)
            .where(BooksModel.id == id)
            .values(**book.model_dump())
            .returning(BooksModel)
        )

        result = await session.execute(query)
        await session.commit()

        updated_book = result.scalar_one_or_none()
        if not updated_book:
            raise HTTPException(status_code = status.HTTP_404_NOT_FOUND, detailt = 'Book not found')
        return updated_book
    @classmethod
    async def del_book(cls, id: int, session: AsyncSession):
        book = await session.get(BooksModel, id)

        if not book:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Book not found')
        await session.delete(book)
        await session.commit()