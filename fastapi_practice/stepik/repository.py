from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.task import TasksModel
from schemas.task import STaskAdd
class TaskRepository:
    @classmethod
    async def add_one(cls, data: STaskAdd, session: AsyncSession) -> TasksModel:
        task_dict = data.model_dump()
        
        task = TasksModel(**task_dict)
        
        session.add(task)
        await session.commit()
        await session.refresh(task)
        
        return task

    @classmethod
    async def find_all(cls, session: AsyncSession):
        query = select(TasksModel)
        
        result = await session.execute(query)
        
        tasks_models = result.scalars().all()
        return tasks_models