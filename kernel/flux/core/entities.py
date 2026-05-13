from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum

class EntityType(Enum):
    MODEL_ENSEMBLE = "Model.Ensemble"
    MODEL_TRANSFORMER = "Model.Transformer"
    HUMAN_IN_THE_LOOP = "Human.In.The.Loop"
    ECOSYSTEM = "ECOSYSTEM"
    RECURSIVE_ENSEMBLE = "Recursive.Ensemble"

@dataclass
class Entity:
    name: str
    type: EntityType
    nature: str = ""
    quantum_state: str = "4bit"
    cost: float = 0.0
    fallback: Optional[str] = None
    model_key: Optional[str] = None
    recursive_config: Optional[Dict[str, Any]] = None
    human_config: Optional[Dict[str, Any]] = None

@dataclass
class Stage:
    name: str
    entity_name: str
    temperature: float = 0.7
    max_new_tokens: int = 1024
    prompt_transform: Optional[str] = None
    observe: Optional[str] = None
    ground_truth: Optional[str] = None

@dataclass
class Attractor:
    name: str
    on_intent: List[str] = field(default_factory=list)
    stages: List[Stage] = field(default_factory=list)

@dataclass
class Ecosystem:
    name: str
    entities: Dict[str, Entity] = field(default_factory=dict)
    attractors: Dict[str, Attractor] = field(default_factory=dict)
    default_policy: str = "Explore & Consolidate"
    observability: str = "Complete"
    policy: Dict = field(default_factory=dict)
