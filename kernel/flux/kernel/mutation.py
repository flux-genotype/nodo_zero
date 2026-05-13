import copy
import os
from datetime import datetime, timezone
from flux.core.entities import Ecosystem, Attractor, Stage
from flux.core.constants import PHI
from flux.parser.flux_serializer import serialize_ecosystem
import json

class MutationEngine:
    def __init__(self, ecosystem: Ecosystem, logger, output_dir: str = "./mutations"):
        self.ecosystem = ecosystem
        self.logger = logger
        self.output_dir = output_dir
        self.mutation_history = {}
        os.makedirs(self.output_dir, exist_ok=True)

    def propose_mutation(self, attractor: Attractor, confidence_score: float, metrics: dict = None):
        mutated_eco = copy.deepcopy(self.ecosystem)
        mutated_attr = mutated_eco.attractors[attractor.name]

        mutation_type = "adjust_temperature"
        if confidence_score < 0.5:
            for stage in mutated_attr.stages:
                stage.temperature = min(2.0, stage.temperature * PHI)
            mutation_type = "explore_increase_temperature"
        elif confidence_score < 0.7:
            for stage in mutated_attr.stages:
                stage.temperature = min(1.5, stage.temperature * 1.1)
            mutation_type = "fine_tune_temperature"
        else:
            for stage in mutated_attr.stages:
                stage.temperature = max(0.1, stage.temperature / PHI)
            mutation_type = "consolidate_reduce_temperature"

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"{attractor.name}_mutated_{timestamp}.flux")
        
        content = f"// Proposed mutation for {attractor.name}\n"
        content += f"// Confidence: {confidence_score:.3f}, Type: {mutation_type}\n"
        if metrics:
            content += f"// Metrics: {json.dumps(metrics)}\n"
        content += serialize_ecosystem(mutated_eco)
        
        with open(filename, 'w') as f:
            f.write(content)

        self.logger.log_event("mutation_proposed", {
            "attractor": attractor.name,
            "confidence": confidence_score,
            "mutation_type": mutation_type,
            "mutated_file": filename,
            "metrics": metrics
        })
        self.mutation_history.setdefault(attractor.name, []).append({
            "timestamp": timestamp,
            "confidence": confidence_score,
            "type": mutation_type,
            "file": filename
        })
