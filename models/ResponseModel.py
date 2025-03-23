from pydantic import BaseModel
from typing import List, Union,Any


class ResponseModel(BaseModel):
    code: int = 200
    msg: str = None
    data: Union[List, str, Any] = None
