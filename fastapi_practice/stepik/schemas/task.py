from pydantic import BaseModel, ConfigDict # 1. Импортируем ConfigDict

class STaskBase(BaseModel):
    name: str
    description: str | None = None
    is_completed: bool = False # Добавили, так как в модели БД оно есть

class STaskAdd(STaskBase):
    pass

class STask(STaskBase):
    id: int
    
    # 2. Включаем поддержку ORM
    model_config = ConfigDict(from_attributes=True)