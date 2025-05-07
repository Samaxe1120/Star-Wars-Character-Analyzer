from typing import Any, Dict, List, Optional, Tuple
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field 


class Summary(BaseModel):
    """Summary of the LinkedIn profile."""
    summary: str = Field(description="Summary of the Star Wars character")
    facts: List[str] = Field(description="List of movies the character has appeared in")


def to_dict(self) -> Dict[str, Any]:
    """Convert the object to a dictionary."""
    return {"summary": self.summary, "facts": self.facts}

summary_parser = PydanticOutputParser(pydantic_object=Summary)

