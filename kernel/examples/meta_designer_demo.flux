ECOSYSTEM "MetaDesignerDemo" {
  ENTITY "fast" {
    TYPE: Model.Transformer
    MODEL_KEY: "fast"
    COSTO: 0.001
  }
  ENTITY "accurate" {
    TYPE: Model.Transformer
    MODEL_KEY: "accurate"
    COSTO: 0.002
  }
  ENTITY "judge" {
    TYPE: Model.Transformer
    MODEL_KEY: "judge-model"
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
    MAX_LATENCY_MS: 30000
    JUDGE_ENTITY: "judge"
    REQUIRE_JUDGE: true
  }
}
