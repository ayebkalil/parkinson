import { useEffect, useState } from 'react'
import { FlaskConical, LoaderCircle, RotateCcw } from 'lucide-react'
import { api } from './api'
import type { Benchmark } from './types'

interface Results { items: Benchmark[]; historical: boolean; protocol: string; note: string }
const value = (n: number | null | undefined) => n == null ? '—' : n.toLocaleString('fr-FR', { minimumFractionDigits: 3, maximumFractionDigits: 3 })

export default function BenchmarkResults() {
  const [protocol, setProtocol] = useState<'audited' | 'historical'>('audited')
  const [results, setResults] = useState<Results | null>(null)
  const [horizon, setHorizon] = useState<12 | 24>(12)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(true)
  const [retry, setRetry] = useState(0)
  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout> | undefined
    setBusy(true); setError('')
    api<Results>(`/benchmarks?protocol=${protocol}`, { signal: controller.signal }).then(data => {
      setResults(data)
      if (protocol === 'audited' && data.historical) timer = setTimeout(() => setRetry(v => v + 1), 10000)
    }).catch(e => { if (!controller.signal.aborted) setError(e.message) }).finally(() => { if (!controller.signal.aborted) setBusy(false) })
    return () => { controller.abort(); clearTimeout(timer) }
  }, [protocol, retry])
  const audited = results && !results.historical
  const rows = results?.items.filter(r => r.target_horizon.includes(`+${horizon} Months`) && !r.model_name.includes('Classifier')).sort((a, b) => (a.mae ?? Infinity) - (b.mae ?? Infinity)) ?? []
  return <section className="card benchmark-results">
    <div className="card-header"><div>
      <h3>{audited ? 'Comparer les expériences auditées' : 'Consulter les archives historiques'}</h3>
      <p>{audited ? 'Protocole corrigé · moyenne de 3 graines · sélection sur validation' : 'Les scores archivés ne permettent pas de mesurer les nouveaux gains.'}</p>
    </div><div className="benchmark-controls">
      <div className="segmented" aria-label="Source des résultats"><button className={protocol === 'audited' ? 'selected' : ''} onClick={() => setProtocol('audited')}>Audité</button><button className={protocol === 'historical' ? 'selected' : ''} onClick={() => setProtocol('historical')}>Archive</button></div>
      <div className="segmented" aria-label="Horizon de prédiction">{([12, 24] as const).map(h => <button key={h} className={horizon === h ? 'selected' : ''} onClick={() => setHorizon(h)}>+{h} mois</button>)}</div>
    </div></div>
    <div className="benchmark-notice"><FlaskConical size={16} /><p>{results?.note ?? 'Chargement des résultats calculés…'}</p></div>
    {error && <div className="error-banner" role="alert">{error}<button onClick={() => setRetry(v => v + 1)}><RotateCcw size={13} /> Réessayer</button></div>}
    {busy && !results && <div className="benchmark-loading" role="status"><LoaderCircle size={18} className="spinning" /> Lecture des résultats…</div>}
    <div className="table-scroll"><table className="benchmark-table"><thead><tr>
      <th>Modèle</th><th>Patients</th>{audited && <th>MAE avant</th>}<th>{audited ? 'MAE retenue ± écart type' : 'MAE ↓'}</th>{audited && <th>Δ MAE</th>}<th>RMSE ↓</th><th>R² ↑</th><th>Pearson r ↑</th>
    </tr></thead><tbody>{rows.map((row, i) => <tr key={row.model_name}>
      <td><span className="rank">{String(i + 1).padStart(2, '0')}</span>{row.model_name.replace('Model 1: ', '').replace('Model 2: ', '').replace('Model 3: ', '').replace('Fusion: ', '')}</td>
      <td>{row.n_samples}</td>{audited && <td>{value(row.baseline_mae)}</td>}
      <td><strong>{value(row.mae)}</strong>{audited && <span className="benchmark-std"> ± {value(row.mae_std)}</span>}</td>
      {audited && <td className={(row.delta_mae ?? 0) < 0 ? 'mae-improved' : (row.delta_mae ?? 0) > 0 ? 'mae-increased' : ''}>{(row.delta_mae ?? 0) > 0 ? '+' : ''}{value(row.delta_mae)}</td>}
      <td>{value(row.rmse)}</td><td>{value(row.r2)}</td><td>{value(row.pearson_r)}</td>
    </tr>)}</tbody></table></div>
    {audited && <p className="benchmark-footnote">Δ MAE = configuration retenue − référence corrigée ; une valeur négative indique une baisse. Les intervalles bootstrap appariés et les résultats par graine figurent dans le rapport d’expérience. Les modèles actifs du dashboard restent distincts de ces expériences.</p>}
  </section>
}
