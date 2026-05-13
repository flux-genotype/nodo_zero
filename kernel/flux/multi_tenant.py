from dataclasses import dataclass, field
from typing import Dict, Optional
from flux.core.entities import Ecosystem
from flux.backends.base import AbstractBackend

@dataclass
class TenantConfig:
    tenant_id: str
    ecosystem: Ecosystem
    backend: AbstractBackend
    model_mapping: Dict[str, str] = field(default_factory=dict)
    budgets: Dict[str, float] = field(default_factory=dict)
    max_parallel_requests: int = 5

class TenantManager:
    def __init__(self):
        self._tenants: Dict[str, TenantConfig] = {}

    def register_tenant(self, config: TenantConfig):
        self._tenants[config.tenant_id] = config

    def get_tenant(self, tenant_id: str) -> TenantConfig:
        if tenant_id not in self._tenants:
            raise ValueError(f"Tenant '{tenant_id}' not found")
        return self._tenants[tenant_id]

    def list_tenants(self) -> list:
        return list(self._tenants.keys())
