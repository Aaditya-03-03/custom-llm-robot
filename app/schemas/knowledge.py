from typing import List, Optional
from pydantic import BaseModel, Field

class KnowledgeQueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question regarding project or robot documentation",
        json_schema_extra={"example": "What platform is used for robot interaction?"}
    )

class DocumentSource(BaseModel):
    document: str = Field(..., description="Source document filename", json_schema_extra={"example": "robot_info.md"})
    chunk_id: Optional[str] = Field(None, description="Specific text chunk identifier", json_schema_extra={"example": "robot_info_md_chunk_0"})
    score: Optional[float] = Field(None, description="Similarity relevance score (0.0 to 1.0)", json_schema_extra={"example": 0.82})

class KnowledgeQueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer based on project knowledge base context")
    sources: List[DocumentSource] = Field(default_factory=list, description="List of source document chunks referenced")
