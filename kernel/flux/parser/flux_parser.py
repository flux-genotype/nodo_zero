from lark import Lark, Transformer, v_args, Tree
from flux.core.entities import Ecosystem, Entity, Stage, Attractor, EntityType

_FLUX_GRAMMAR = r'''
ECOSYSTEM_KW: "ECOSYSTEM"
ENTITY_KW: "ENTITY"
ATTRACTOR_KW: "ATTRACTOR"
STAGE_KW: "STAGE"
POLICY_KW: "POLICY"
RECURSIVE_CONFIG_KW: "RECURSIVE_CONFIG"
HUMAN_CONFIG_KW: "HUMAN_CONFIG"
LBRACE: "{"
RBRACE: "}"
LBRACKET: "["
RBRACKET: "]"
COMMA: ","

start: ecosystem

ecosystem: ECOSYSTEM_KW ESCAPED_STRING LBRACE [ecosystem_body] RBRACE
ecosystem_body: (entity_def | attractor_def | default_policy | osservabilita | policy_def)*

entity_def: ENTITY_KW ESCAPED_STRING LBRACE [entity_body] RBRACE
entity_body: (type_def | nature_def | quantum_state_def | costo_def | fallback_def | model_key_def | recursive_config_def | human_config_def)*

attractor_def: ATTRACTOR_KW ESCAPED_STRING LBRACE attractor_body RBRACE
attractor_body: on_intent_def stage_def+

stage_def: STAGE_KW ESCAPED_STRING LBRACE [stage_body] RBRACE
stage_body: (execute_def | temperature_def | max_new_tokens_def | prompt_transform_def | observe_def | ground_truth_def)*

type_def: "TYPE:" entity_type
entity_type: ENTITY_TYPE
ENTITY_TYPE: "Model.Ensemble" | "Model.Transformer" | "Human.In.The.Loop" | "ECOSYSTEM" | "Recursive.Ensemble"

nature_def: "NATURE:" ESCAPED_STRING
quantum_state_def: "QUANTUM_STATE:" ESCAPED_STRING
costo_def: "COSTO:" NUMBER
fallback_def: "FALLBACK:" ESCAPED_STRING
model_key_def: "MODEL_KEY:" ESCAPED_STRING

recursive_config_def: RECURSIVE_CONFIG_KW LBRACE [recursive_config_body] RBRACE
recursive_config_body: (max_chunk_tokens_def | chunk_overlap_def | sub_model_key_def | root_model_key_def | max_workers_def)*
max_chunk_tokens_def: "MAX_CHUNK_TOKENS:" NUMBER
chunk_overlap_def: "CHUNK_OVERLAP:" NUMBER
sub_model_key_def: "SUB_MODEL_KEY:" ESCAPED_STRING
root_model_key_def: "ROOT_MODEL_KEY:" ESCAPED_STRING
max_workers_def: "MAX_WORKERS:" NUMBER

human_config_def: HUMAN_CONFIG_KW LBRACE [human_config_body] RBRACE
human_config_body: (human_timeout_def | human_escalation_def | human_confidence_threshold_def)*
human_timeout_def: "TIMEOUT_SEC:" NUMBER
human_escalation_def: "ESCALATION:" ESCAPED_STRING
human_confidence_threshold_def: "CONFIDENCE_THRESHOLD:" NUMBER

on_intent_def: "ON_INTENT:" LBRACKET [string_list] RBRACKET
string_list: ESCAPED_STRING (COMMA ESCAPED_STRING)*

execute_def: "EXECUTE:" ESCAPED_STRING
temperature_def: "TEMPERATURE:" NUMBER
max_new_tokens_def: "MAX_NEW_TOKENS:" NUMBER
prompt_transform_def: "PROMPT_TRANSFORM:" ESCAPED_STRING
observe_def: "OBSERVE:" ESCAPED_STRING
ground_truth_def: "GROUND_TRUTH:" ESCAPED_STRING

default_policy: "DEFAULT_POLICY:" ESCAPED_STRING
osservabilita: "OSSERVABILITÀ:" ESCAPED_STRING

policy_def: POLICY_KW LBRACE [policy_body] RBRACE
policy_body: (max_cost | max_latency | max_human_iterations | judge_entity | require_judge)*
max_cost: "MAX_COST_PER_REQUEST:" NUMBER
max_latency: "MAX_LATENCY_MS:" NUMBER
max_human_iterations: "MAX_HUMAN_ITERATIONS:" NUMBER
judge_entity: "JUDGE_ENTITY:" ESCAPED_STRING
require_judge: "REQUIRE_JUDGE:" ("true" | "false")

%import common.ESCAPED_STRING
%import common.SIGNED_NUMBER -> NUMBER
%import common.WS
%ignore WS
%ignore /\/\/[^\n]*/
%ignore /#[^\n]*/
'''

@v_args(inline=True)
class FluxTransformer(Transformer):
    def __init__(self):
        super().__init__()

    def start(self, ecosystem):
        return ecosystem

    # ------------------------------------------------------------
    # Discard literal tokens (keywords and braces) – return None
    # ------------------------------------------------------------
    def ECOSYSTEM_KW(self, _): return None
    def ENTITY_KW(self, _): return None
    def ATTRACTOR_KW(self, _): return None
    def STAGE_KW(self, _): return None
    def POLICY_KW(self, _): return None
    def RECURSIVE_CONFIG_KW(self, _): return None
    def HUMAN_CONFIG_KW(self, _): return None
    def LBRACE(self, _): return None
    def RBRACE(self, _): return None
    def LBRACKET(self, _): return None
    def RBRACKET(self, _): return None
    def COMMA(self, _): return None

    # Terminals
    def ESCAPED_STRING(self, s):
        return s[1:-1]

    def NUMBER(self, n):
        return float(n)

    # ------------------------------------------------------------
    # Helper: extract the useful value from *args (last non None)
    # ------------------------------------------------------------
    def _extract_value(self, args):
        args = [a for a in args if a is not None]
        return args[-1] if args else None

    # ------------------------------------------------------------
    # Main methods (iterate over body.children)
    # ------------------------------------------------------------
    def ecosystem(self, _ecosystem, name, _lbrace, body, _rbrace):
        eco = Ecosystem(name=name)
        if body is not None:
            for item in body.children:
                if isinstance(item, Entity):
                    eco.entities[item.name] = item
                elif isinstance(item, Attractor):
                    eco.attractors[item.name] = item
                elif isinstance(item, tuple):
                    if item[0] == 'policy':
                        eco.policy = item[1]
                    elif item[0] == 'default_policy':
                        eco.default_policy = item[1]
                    elif item[0] == 'osservabilita':
                        eco.observability = item[1]
        return eco

    def entity_def(self, _entity, name, _lbrace, body, _rbrace):
        entity = Entity(name=name, type=EntityType.ECOSYSTEM)
        if body is not None:
            for arg in body.children:
                if isinstance(arg, tuple):
                    key, val = arg[0], arg[1]
                    if key == 'type':
                        # val could be Tree (if entity_type has no method)
                        if isinstance(val, Tree):
                            val = str(val.children[0])
                        entity.type = EntityType(val)
                    elif key == 'nature':
                        entity.nature = val
                    elif key == 'quantum_state':
                        entity.quantum_state = val
                    elif key == 'cost':
                        entity.cost = val
                    elif key == 'fallback':
                        entity.fallback = val
                    elif key == 'model_key':
                        entity.model_key = val
                    elif key == 'recursive_config':
                        entity.recursive_config = val
                    elif key == 'human_config':
                        entity.human_config = val
        return entity

    def attractor_def(self, _attractor, name, _lbrace, body, _rbrace):
        on_intent = []
        stages = []
        if body is not None:
            for item in body.children:
                if isinstance(item, list):
                    on_intent = item
                elif isinstance(item, Stage):
                    stages.append(item)
        return Attractor(name=name, on_intent=on_intent, stages=stages)

    def stage_def(self, _stage, name, _lbrace, body, _rbrace):
        stage = Stage(name=name, entity_name="")
        if body is not None:
            for arg in body.children:
                if isinstance(arg, tuple):
                    key, val = arg[0], arg[1]
                    if key == 'execute':
                        stage.entity_name = val
                    elif key == 'temperature':
                        stage.temperature = val
                    elif key == 'max_new_tokens':
                        stage.max_new_tokens = int(val)
                    elif key == 'prompt_transform':
                        stage.prompt_transform = val
                    elif key == 'observe':
                        stage.observe = val
                    elif key == 'ground_truth':
                        stage.ground_truth = val
        return stage

    # ------------------------------------------------------------
    # Methods for definitions with literal tokens → *args
    # ------------------------------------------------------------
    def type_def(self, *args):
        val = self._extract_value(args)
        if isinstance(val, Tree):
            val = str(val.children[0])
        return ('type', str(val))

    def nature_def(self, *args):
        return ('nature', self._extract_value(args))

    def quantum_state_def(self, *args):
        return ('quantum_state', self._extract_value(args))

    def costo_def(self, *args):
        return ('cost', self._extract_value(args))

    def fallback_def(self, *args):
        return ('fallback', self._extract_value(args))

    def model_key_def(self, *args):
        return ('model_key', self._extract_value(args))

    def recursive_config_def(self, *args):
        filtered = [a for a in args if a is not None]
        body = filtered[0] if filtered else None
        config = {}
        if body is not None and hasattr(body, 'children'):
            for item in body.children:
                if isinstance(item, tuple):
                    config[item[0]] = item[1]
        return ('recursive_config', config)

    def max_chunk_tokens_def(self, *args):
        return ('max_chunk_tokens', int(self._extract_value(args)))

    def chunk_overlap_def(self, *args):
        return ('chunk_overlap', int(self._extract_value(args)))

    def sub_model_key_def(self, *args):
        return ('sub_model_key', self._extract_value(args))

    def root_model_key_def(self, *args):
        return ('root_model_key', self._extract_value(args))

    def max_workers_def(self, *args):
        return ('max_workers', int(self._extract_value(args)))

    def human_config_def(self, *args):
        filtered = [a for a in args if a is not None]
        body = filtered[0] if filtered else None
        config = {}
        if body is not None and hasattr(body, 'children'):
            for item in body.children:
                if isinstance(item, tuple):
                    config[item[0]] = item[1]
        return ('human_config', config)

    def human_timeout_def(self, *args):
        return ('timeout_sec', int(self._extract_value(args)))

    def human_escalation_def(self, *args):
        return ('escalation', self._extract_value(args))

    def human_confidence_threshold_def(self, *args):
        return ('confidence_threshold', float(self._extract_value(args)))

    def on_intent_def(self, *args):
        # args: ON_INTENT, LBRACKET, string_list, RBRACKET
        # string_list can be None; look for the first argument of type list
        for a in args:
            if isinstance(a, list):
                return a
        return []

    def string_list(self, first, *rest):
        return [first] + [r for r in rest if r is not None]

    def execute_def(self, *args):
        return ('execute', self._extract_value(args))

    def temperature_def(self, *args):
        return ('temperature', self._extract_value(args))

    def max_new_tokens_def(self, *args):
        return ('max_new_tokens', self._extract_value(args))

    def prompt_transform_def(self, *args):
        return ('prompt_transform', self._extract_value(args))

    def observe_def(self, *args):
        return ('observe', self._extract_value(args))

    def ground_truth_def(self, *args):
        return ('ground_truth', self._extract_value(args))

    def default_policy(self, *args):
        return ('default_policy', self._extract_value(args))

    def osservabilita(self, *args):
        return ('osservabilita', self._extract_value(args))

    def policy_def(self, *args):
        filtered = [a for a in args if a is not None]
        body = filtered[0] if filtered else None
        policy_dict = {}
        if body is not None and hasattr(body, 'children'):
            for item in body.children:
                if isinstance(item, tuple):
                    policy_dict[item[0]] = item[1]
        return ('policy', policy_dict)

    def max_cost(self, *args):
        return ('MAX_COST_PER_REQUEST', self._extract_value(args))

    def max_latency(self, *args):
        return ('MAX_LATENCY_MS', self._extract_value(args))

    def max_human_iterations(self, *args):
        return ('MAX_HUMAN_ITERATIONS', int(self._extract_value(args)))

    def judge_entity(self, *args):
        return ('JUDGE_ENTITY', str(self._extract_value(args)))

    def require_judge(self, *args):
        val = str(self._extract_value(args)).lower() == "true"
        return ('REQUIRE_JUDGE', val)


class FluxParser:
    def __init__(self, grammar_path=None):
        if grammar_path is None:
            self.parser = Lark(_FLUX_GRAMMAR, parser='lalr', transformer=FluxTransformer())
        else:
            with open(grammar_path, 'r') as f:
                grammar = f.read()
            self.parser = Lark(grammar, parser='lalr', transformer=FluxTransformer())

    def parse_file(self, filepath: str) -> Ecosystem:
        with open(filepath, 'r') as f:
            code = f.read()
        return self.parser.parse(code)
