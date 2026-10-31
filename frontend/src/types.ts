export type ModelId = 'late_fusion' | 'bilstm' | 'xgboost'
export type Page = 'dashboard' | 'patients' | 'models' | 'settings'
export type Group = 'all' | 'rapid' | 'stable' | 'history' | 'imaging'
export interface EngineStatus {
  state: 'idle' | 'loading' | 'ready' | 'error'
  progress: number
  stage: string
  loaded_models: string[]
  error: string | null
  ready_at: string | null
}
export interface Patient {
  id: number
  age: number | null
  sex: string
  score: number | null
  stage: number | null
  visits: number
  last_visit: string
  rapid_progression: boolean
  has_outcome: boolean
  putamen_sbr: number | null
  cohort: string
}
export interface Horizon {
  score: number
  delta: number
  index_score: number
  index_month: number
  index_date: string
  month: number
}
export interface Visit {
  date: string
  month: number
  score: number | null
  stage: number | null
  ledd: number | null
  moca: number | null
  putamen_sbr: number | null
  attention: number
}
export interface Prediction {
  patient: Patient
  predictions: Record<ModelId, { '12m': Horizon | null; '24m': Horizon | null; probability: number | null }>
  history: Visit[]
  gating: { clinical: number; imaging: number }
  biomarkers: Record<string, number | null>
  provenance: string
}
export interface Overview {
  patients: number
  visits: number
  models: number
  mean_score: number
  mean_age: number
  rapid_count: number
  rapid_rate: number
  imaging_count: number
  ready_at: string
  trend: { month: number; score: number; n: number }[]
  stages: { stage: string; count: number }[]
}
export interface Benchmark {
  model_name: string
  target_horizon: string
  n_samples: number
  mae: number | null
  rmse: number | null
  r2: number | null
  pearson_r: number | null
  notes: string
  baseline_mae?: number
  delta_mae?: number
  mae_std?: number
  n_seeds?: number
}
