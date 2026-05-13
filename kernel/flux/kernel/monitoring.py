from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest, REGISTRY

class NoopMetrics:
    def record_request(self, *args, **kwargs): pass
    def set_confidence(self, *args, **kwargs): pass
    def set_stage_temperature(self, *args, **kwargs): pass
    def add_cost(self, *args, **kwargs): pass
    def set_meta_designer_score(self, phase: str, value: float):
        pass  # aggiunto per uniformità
    def get_metrics(self) -> bytes:
        return b""

class FluxMetrics:
    _shared_instance = None
    _shared_registry = CollectorRegistry()

    def __init__(self, registry: CollectorRegistry = None):
        if registry is None:
            registry = FluxMetrics._shared_registry
        self.registry = registry
        self.requests_total = Counter(
            'flux_requests_total', 'Total number of processed requests',
            ['tenant_id', 'attractor', 'stage'], registry=registry
        )
        self.errors_total = Counter(
            'flux_errors_total', 'Total errors (non-transient)',
            ['tenant_id', 'attractor', 'stage', 'error_type'], registry=registry
        )
        self.request_duration = Histogram(
            'flux_request_duration_seconds', 'Request latency (seconds)',
            ['tenant_id', 'attractor', 'stage'],
            buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60], registry=registry
        )
        self.confidence_gauge = Gauge(
            'flux_confidence_score', 'Current confidence (last run)',
            ['tenant_id', 'attractor'], registry=registry
        )
        self.stage_temperature = Gauge(
            'flux_stage_temperature', 'Effective temperature used in the stage',
            ['tenant_id', 'attractor', 'stage'], registry=registry
        )
        self.cost_total = Counter(
            'flux_cost_total', 'Cumulative cost in monetary units',
            ['tenant_id'], registry=registry
        )
        # ---- Nuovo Gauge per MetaDesigner ----
        self.meta_designer_score = Gauge(
            'flux_meta_designer_score', 'MetaDesigner ecosystem evaluation score',
            ['phase'], registry=registry
        )
        # ------------------------------------
        self.info = Info('flux_build', 'Information about FLUX Kernel version', registry=registry)
        self.info.info({'version': '2.2.0'})

    @classmethod
    def get_shared(cls):
        if cls._shared_instance is None:
            cls._shared_instance = cls()
        return cls._shared_instance

    def record_request(self, tenant_id: str, attractor: str, stage: str, duration: float, success: bool, error: str = None):
        self.requests_total.labels(tenant_id=tenant_id, attractor=attractor, stage=stage).inc()
        self.request_duration.labels(tenant_id=tenant_id, attractor=attractor, stage=stage).observe(duration)
        if not success and error:
            self.errors_total.labels(tenant_id=tenant_id, attractor=attractor, stage=stage, error_type=error).inc()

    def set_confidence(self, tenant_id: str, attractor: str, confidence: float):
        self.confidence_gauge.labels(tenant_id=tenant_id, attractor=attractor).set(confidence)

    def set_stage_temperature(self, tenant_id: str, attractor: str, stage: str, temperature: float):
        self.stage_temperature.labels(tenant_id=tenant_id, attractor=attractor, stage=stage).set(temperature)

    def add_cost(self, tenant_id: str, amount: float):
        self.cost_total.labels(tenant_id=tenant_id).inc(amount)

    def set_meta_designer_score(self, phase: str, value: float):
        self.meta_designer_score.labels(phase=phase).set(value)

    def get_metrics(self) -> bytes:
        return generate_latest(self.registry)