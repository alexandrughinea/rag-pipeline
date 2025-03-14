from typing import Optional

from pydantic import BaseModel

default_n_results = 5

class QueryBaseRequest(BaseModel):
    query: str
    n_results: Optional[int] = default_n_results

class QueryConversationRequest(BaseModel):
    query: str
    conversation_id: Optional[int] = None
    n_results: Optional[int] = default_n_results
