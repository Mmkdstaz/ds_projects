from fastapi import APIRouter, HTTPException, status
from schemas.task import STask, STaskAdd
from database import SessionDep
from typing import Annotated
from sqlalchemy import select
from models.task import TasksModel
router = APIRouter(prefix="/tasks", tags=["Задачи"])

tasks = []


@router.post("", response_model=STask)  
async def create_task(task: STaskAdd, session: SessionDep):
    new_task = TasksModel(**task.model_dump())
    session.add(new_task)
    await session.commit()
    await session.refresh(new_task)
    return new_task


@router.get("")
async def get_tasks(session: SessionDep):
    query = select(TasksModel)
    result = await session.execute(query)
    return result.scalars().all()

@router.get("/{task_id}")
async def get_task(task_id: int):
    for task in tasks:
        if task['id'] == task_id:
            return task