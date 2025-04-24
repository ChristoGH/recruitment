"""
Module containing data models.
"""
from dataclasses import dataclass

@dataclass
class URL:
    """Class representing a URL."""
    id: int
    url: str
    domain: str
    source: str 