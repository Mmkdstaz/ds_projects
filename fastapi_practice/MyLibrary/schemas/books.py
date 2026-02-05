from pydantic import BaseModel, ConfigDict, Field

class SBookBase(BaseModel):
    title: str
    author: str
    year: int
    pages: int = Field(..., ge = 10)
    is_read: bool = False

class SBookAdd(SBookBase):
    pass

class SBook(SBookAdd):
    id: int

    model_config = ConfigDict(from_attributes = False)