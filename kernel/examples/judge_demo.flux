ECOSYSTEM "LLM Judge Demo" {
  ENTITY "fast" {
    TYPE: Model.Transformer
    MODEL_KEY: "fast"
    COSTO: 0.001
  }
  ENTITY "judge" {
    TYPE: Model.Transformer
    MODEL_KEY: "judge-model"
    COSTO: 0.002
  }
  ENTITY "architect" {
    TYPE: Model.Transformer
    MODEL_KEY: "architect"
    COSTO: 0.002
  }
  ATTRACTOR "qa" {
    ON_INTENT: ["*"]
    STAGE "Answer" {
      EXECUTE: "fast"
      TEMPERATURE: 0.7
      MAX_NEW_TOKENS: 512
      OBSERVE: "Answer the question."
    }
  }
  POLICY {
    MAX_COST_PER_REQUEST: 0.05
    MAX_LATENCY_MS: 1000000
    JUDGE_ENTITY: "judge"
    REQUIRE_JUDGE: true
  }
}
