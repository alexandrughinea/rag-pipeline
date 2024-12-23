from typing import Optional

from pydantic import BaseModel


# Define request model
class QueryBaseRequest(BaseModel):
    query: str

# Define request model
class QueryConversationRequest(BaseModel):
    query: str
    conversation_id: Optional[int] = None
