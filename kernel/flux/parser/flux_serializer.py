from flux.core.entities import Ecosystem, Entity, Attractor, Stage

def serialize_ecosystem(eco: Ecosystem) -> str:
    lines = [f'ECOSYSTEM "{eco.name}" {{']

    for ent in eco.entities.values():
        lines.append(f' ENTITY "{ent.name}" {{')
        lines.append(f' TYPE: {ent.type.value}')
        if ent.nature:
            lines.append(f' NATURE: "{ent.nature}"')
        lines.append(f' QUANTUM_STATE: "{ent.quantum_state}"')
        lines.append(f' COSTO: {ent.cost}')
        if ent.fallback:
            lines.append(f' FALLBACK: "{ent.fallback}"')
        if ent.model_key:
            lines.append(f' MODEL_KEY: "{ent.model_key}"')
        if ent.recursive_config:
            lines.append(' RECURSIVE_CONFIG {')
            for k, v in ent.recursive_config.items():
                if isinstance(v, str):
                    lines.append(f' {k.upper()}: "{v}"')
                else:
                    lines.append(f' {k.upper()}: {v}')
            lines.append(' }')
        if ent.human_config:
            lines.append(' HUMAN_CONFIG {')
            for k, v in ent.human_config.items():
                if isinstance(v, str):
                    lines.append(f' {k.upper()}: "{v}"')
                else:
                    lines.append(f' {k.upper()}: {v}')
            lines.append(' }')
        lines.append(' }')

    for attr in eco.attractors.values():
        lines.append(f' ATTRACTOR "{attr.name}" {{')
        intents = ', '.join(f'"{i}"' for i in attr.on_intent)
        lines.append(f' ON_INTENT: [{intents}]')
        for stage in attr.stages:
            lines.append(f' STAGE "{stage.name}" {{')
            lines.append(f' EXECUTE: "{stage.entity_name}"')
            if stage.temperature != 0.7:
                lines.append(f' TEMPERATURE: {stage.temperature}')
            if stage.max_new_tokens != 1024:
                lines.append(f' MAX_NEW_TOKENS: {stage.max_new_tokens}')
            if stage.prompt_transform:
                lines.append(f' PROMPT_TRANSFORM: "{stage.prompt_transform}"')
            if stage.observe:
                lines.append(f' OBSERVE: "{stage.observe}"')
            if stage.ground_truth:
                lines.append(f' GROUND_TRUTH: "{stage.ground_truth}"')
            lines.append(' }')
        lines.append(' }')

    if eco.policy:
        lines.append(' POLICY {')
        for k, v in eco.policy.items():
            if isinstance(v, bool):
                lines.append(f' {k}: {"true" if v else "false"}')
            elif isinstance(v, str):
                lines.append(f' {k}: "{v}"')
            else:
                lines.append(f' {k}: {v}')
        lines.append(' }')

    if eco.default_policy:
        lines.append(f' DEFAULT_POLICY: "{eco.default_policy}"')
    if eco.observability:
        lines.append(f' OSSERVABILITÀ: "{eco.observability}"')

    lines.append('}')
    return '\n'.join(lines)

def deserialize_ecosystem_file(filepath: str) -> Ecosystem:
    from flux.parser import FluxParser
    parser = FluxParser()
    return parser.parse_file(filepath)
