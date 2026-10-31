import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react'
import type { ComponentType, ReactNode } from 'react'
import {
  Activity, ArrowRight, ArrowUpRight, AudioLines, BrainCircuit,
  ChartNoAxesCombined, Check, CheckCheck, ChevronDown, ChevronLeft, ChevronRight,
  CircleHelp, CirclePlay, Cpu, Download, Expand, FlaskConical, History, LayoutDashboard,
  LoaderCircle, Menu, Microscope, PanelLeftClose, Search, Settings2,
  ShieldCheck, SlidersHorizontal, Sparkles, Stethoscope, TrendingUp, Users, Waves, X,
} from 'lucide-react'
import CinematicIntro from './CinematicIntro'
import { api } from './api'
import BenchmarkResults from './BenchmarkResults'
import type { EngineStatus, Group, ModelId, Overview, Page, Patient, Prediction } from './types'

const TrajectoryPlot = lazy(() => import('./TrajectoryPlot'))

const models: { id: ModelId; name: string; short: string; kind: string; description: string; icon: typeof BrainCircuit }[] = [
  { id: 'late_fusion', name: 'Fusion multimodale', short: 'Late Fusion', kind: 'Fusion adaptative', description: 'Croise les trajectoires cliniques, les biomarqueurs et l’imagerie avec une pondération apprise.', icon: BrainCircuit },
  { id: 'bilstm', name: 'Trajectoire BiLSTM', short: 'BiLSTM', kind: 'Séquences longitudinales', description: 'Analyse l’historique des visites et leur espacement avec un mécanisme d’attention temporelle.', icon: Waves },
  { id: 'xgboost', name: 'XGBoost', short: 'XGBoost', kind: 'Visites index alignées', description: 'Exploite un instantané multimodal à une visite éligible propre à chaque horizon.', icon: ChartNoAxesCombined },
]
const groups: { id: Group; label: string }[] = [
  { id: 'all', label: 'Tous les patients' }, { id: 'rapid', label: 'Progression observée ≥ 3,5 pts' },
  { id: 'stable', label: 'Progression observée < 3,5 pts' }, { id: 'history', label: 'Historique ≥ 4 visites' },
  { id: 'imaging', label: 'DaTSCAN SBR < 1,0' },
]
const fmt = (n: number | null | undefined, digits = 1) => n == null ? '—' : n.toLocaleString('fr-FR', { maximumFractionDigits: digits, minimumFractionDigits: digits })
const integer = (n: number) => n.toLocaleString('fr-FR')
const date = (value: string) => new Date(`${value}T12:00:00`).toLocaleDateString('fr-FR', { day: '2-digit', month: 'short', year: 'numeric' })
const delta = (n: number) => `${n > 0 ? '+' : ''}${fmt(n)} pts`
const initialStatus: EngineStatus = { state: 'idle', progress: 0, stage: 'Connexion à votre espace', loaded_models: [], error: null, ready_at: null }

function Brand({ light = false }: { light?: boolean }) {
  return <div className={`brand ${light ? 'brand-light' : ''}`}><span className="brand-mark"><BrainCircuit size={23} strokeWidth={1.7} /></span><div><strong>neurovia<span>.</span></strong><small>INTELLIGENCE PARKINSON</small></div></div>
}

function Card({ children, className = '' }: { children: ReactNode; className?: string }) { return <section className={`card ${className}`}>{children}</section> }
function Pill({ children, tone = 'purple' }: { children: ReactNode; tone?: string }) { return <span className={`pill pill-${tone}`}>{children}</span> }
function SectionTitle({ eyebrow, title, children }: { eyebrow?: string; title: string; children?: ReactNode }) { return <div className="section-title"><div>{eyebrow && <span className="eyebrow">{eyebrow}</span>}<h2>{title}</h2></div>{children}</div> }
function Empty({ children }: { children: ReactNode }) { return <div className="empty-state"><Search size={28} /><p>{children}</p></div> }

function Metric({ label, value, unit, icon: Icon, tone, detail, spark }: { label: string; value: string; unit?: string; icon: ComponentType<{ size?: number }>; tone: string; detail: string; spark?: number[] }) {
  const trend = spark?.filter(Number.isFinite) ?? []
  const low = Math.min(...trend), span = Math.max(...trend) - low || 1
  return <Card className={`metric-card tone-${tone}`}><div className="metric-top"><span className="metric-icon"><Icon size={20} /></span><span className="metric-kicker">{label.includes('Prévision') ? <><Sparkles size={12} /> PRÉDICTION</> : <><span className="tiny-dot" /> OBSERVÉ</>}</span></div><div className="metric-value">{value}<small>{unit}</small></div><div className="metric-bottom"><div><h3>{label}</h3><span>{detail}</span></div><svg viewBox="0 0 80 32" aria-hidden="true"><polyline points={trend.map((n, i) => `${i * 80 / Math.max(1, trend.length - 1)},${28 - (n - low) / span * 24}`).join(' ')} /></svg></div></Card>
}

function Trajectory({ result, model }: { result: Prediction; model: ModelId }) {
  const [wide, setWide] = useState(false)
  const prediction = result.predictions[model]
  const [horizon, setHorizon] = useState<12 | 24>(24)
  const lastMonth = result.history.at(-1)?.month ?? 0
  const points = useMemo(() => {
    const rows: Record<string, number | null>[] = result.history.map(v => ({ month: v.month, observed: v.score }))
    for (const months of [12, 24] as const) {
      if (months > horizon) continue
      const p = prediction[`${months}m`]
      if (!p) continue
      const key = `forecast${months}`
      for (const [month, value] of [[p.index_month, p.index_score], [p.month, p.score]]) {
        let row = rows.find(r => Math.abs((r.month ?? 0) - month) < 0.01)
        if (!row) { row = { month }; rows.push(row) }
        row[key] = value
      }
    }
    return rows.sort((a, b) => (a.month ?? 0) - (b.month ?? 0))
  }, [result, prediction, horizon])
  return <Card className={`trajectory-card ${wide ? 'chart-expanded' : ''}`}><div className="card-header"><div><h3>Trajectoire & prévisions</h3><p>Évolution du score moteur MDS-UPDRS III</p></div><div className="chart-actions"><div className="segmented">{([12, 24] as const).map(h => <button key={h} onClick={() => setHorizon(h)} className={horizon === h ? 'selected' : ''}>{h} mois</button>)}</div><button className="icon-button" title={wide ? 'Réduire le graphique' : 'Agrandir le graphique'} aria-label={wide ? 'Réduire le graphique' : 'Agrandir le graphique'} onClick={() => setWide(!wide)}>{wide ? <X size={17} /> : <Expand size={17} />}</button></div></div>
    <div className="chart-legend"><span><i className="legend-dot purple" /> Visites observées</span><span><i className="legend-dash" /> Prévision {models.find(m => m.id === model)?.short}</span></div>
    <div className="trajectory-chart" role="img" aria-label="Historique des scores moteurs et prévisions à 12 et 24 mois"><Suspense fallback={<div className="chart-loading" role="status"><LoaderCircle size={20} className="spinning" /> Chargement du graphique...</div>}><TrajectoryPlot points={points} lastMonth={lastMonth} /></Suspense></div>
    <div className="chart-footer"><span><History size={13} /> {result.history.length} visites avant la date index</span><span>Temps depuis la visite initiale · mois</span></div>
    {model === 'xgboost' && <p className="chart-note">Chaque horizon utilise sa propre visite index éligible. Les dates figurent dans les prévisions.</p>}
  </Card>
}

function Attribution({ result, model }: { result: Prediction; model: ModelId }) {
  const isFusion = model === 'late_fusion'
  return <Card className="attribution-card"><div className="card-header"><div><h3>{isFusion ? 'Au cœur de la prédiction' : model === 'bilstm' ? 'Attention temporelle' : 'Visites index du modèle'}</h3><p>{isFusion ? 'Pondération des modalités · Late Fusion' : model === 'bilstm' ? 'Poids appris pour chaque visite' : 'Ancrage propre à chaque horizon'}</p></div><span className="soft-icon"><BrainCircuit size={19} /></span></div>
    {isFusion ? <><div className="gating-visual"><svg viewBox="0 0 170 170" role="img" aria-label={`Clinique ${fmt(result.gating.clinical)} %, imagerie ${fmt(result.gating.imaging)} %`}><circle cx="85" cy="85" r="66" className="gating-bg" /><circle cx="85" cy="85" r="66" className="gating-ring" strokeDasharray={`${result.gating.clinical / 100 * 414.69} 414.69`} /><circle cx="85" cy="85" r="51" className="gating-inner" /></svg><div><BrainCircuit size={27} /><strong>2 modalités</strong><span>une prédiction</span></div></div><div className="gating-labels"><div><span><i className="legend-dot purple" /> Trajectoire clinique</span><strong>{fmt(result.gating.clinical)}<small>%</small></strong></div><div><span><i className="legend-dot teal" /> Imagerie cérébrale</span><strong>{fmt(result.gating.imaging)}<small>%</small></strong></div></div><p className="explain-note">Pondération apprise par le modèle ; elle ne mesure pas une relation causale.</p></> : model === 'bilstm' ? <><div className="attention-bars">{result.history.map((v, i) => <div key={`${v.date}-${i}`} title={`${date(v.date)} · ${fmt(v.attention)} %`}><i style={{ height: `${Math.max(4, v.attention / Math.max(...result.history.map(x => x.attention)) * 100)}%` }} /><small>{i + 1}</small></div>)}</div><p className="explain-note">Les barres représentent l’attention relative portée à chaque visite, de la plus ancienne à la plus récente.</p></> : <div className="index-visits">{(['12m', '24m'] as const).map(h => { const p = result.predictions.xgboost[h]; return <div key={h}><span>Horizon +{h === '12m' ? '12' : '24'} mois</span><strong>{p ? date(p.index_date) : 'Non disponible'}</strong><small>{p ? `Score index : ${fmt(p.index_score)} pts` : 'Aucune visite index éligible'}</small></div> })}<p className="explain-note">Les checkpoints retenus et leur intégrité sont vérifiés au démarrage.</p></div>}
  </Card>
}

function PatientTable({ patients, selected, onSelect, compact = false }: { patients: Patient[]; selected: number; onSelect: (id: number) => void; compact?: boolean }) {
  const [page, setPage] = useState(0)
  const perPage = compact ? 5 : 10
  useEffect(() => setPage(0), [patients])
  const rows = patients.slice(page * perPage, (page + 1) * perPage)
  return <><div className="table-scroll"><table className="patient-table"><thead><tr><th>Patient</th><th>Score moteur</th><th>Dernière visite</th>{!compact && <th>Historique</th>}<th>Évolution observée à 12 m</th><th><span className="sr-only">Ouvrir</span></th></tr></thead><tbody>{rows.map(p => <tr key={p.id} className={p.id === selected ? 'row-selected' : ''}><td><button className="patient-cell" onClick={() => onSelect(p.id)}><span className={`patient-avatar avatar-${p.id % 4}`}>{String(p.id).slice(-2)}</span><span><strong>Patient #{p.id}</strong><small>{fmt(p.age, 0)} ans · {p.sex}</small></span></button></td><td><span className="score-cell">{fmt(p.score)} <small>/ 132</small></span></td><td>{date(p.last_visit)}</td>{!compact && <td>{p.visits} visites</td>}<td>{p.has_outcome ? <Pill tone={p.rapid_progression ? 'amber' : 'teal'}><span className="tiny-dot" />{p.rapid_progression ? '≥ 3,5 points' : '< 3,5 points'}</Pill> : <Pill tone="neutral">Non renseignée</Pill>}</td><td><button className="row-open" aria-label={`Ouvrir le patient ${p.id}`} onClick={() => onSelect(p.id)}><ArrowUpRight size={17} /></button></td></tr>)}</tbody></table></div>{!rows.length && <Empty>Aucun patient ne correspond à cette recherche.</Empty>}<div className="table-pagination"><span>{patients.length ? `${page * perPage + 1}–${Math.min((page + 1) * perPage, patients.length)} sur ${integer(patients.length)} patients` : '0 patient'}</span><div><button className="icon-button" aria-label="Page précédente" disabled={page === 0} onClick={() => setPage(page - 1)}><ChevronLeft size={16} /></button><span>{page + 1} / {Math.max(1, Math.ceil(patients.length / perPage))}</span><button className="icon-button" aria-label="Page suivante" disabled={(page + 1) * perPage >= patients.length} onClick={() => setPage(page + 1)}><ChevronRight size={16} /></button></div></div></>
}

function ModelPage({ active, onChoose }: { active: ModelId; onChoose: (m: ModelId) => void }) {
  return <><div className="model-cards">{models.map((m, i) => <Card key={m.id} className={`model-card ${active === m.id ? 'model-selected' : ''}`}><div className={`model-symbol model-symbol-${i}`}><m.icon size={28} /></div><Pill tone="neutral">{m.kind}</Pill><h3>{m.name}</h3><p>{m.description}</p><div className="model-card-bottom"><span><span className="tiny-dot" /> Chargé et disponible</span><button className={active === m.id ? 'text-button selected-model' : 'text-button'} onClick={() => onChoose(m.id)}>{active === m.id ? 'Modèle actif' : 'Utiliser'}{active === m.id ? <Check size={15} /> : <ArrowRight size={15} />}</button></div></Card>)}</div><BenchmarkResults /></>
}

function HelpDialog({ onClose }: { onClose: () => void }) {
  const closeRef = useRef<HTMLButtonElement>(null)
  useEffect(() => { closeRef.current?.focus(); const close = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); if (e.key === 'Tab') { e.preventDefault(); closeRef.current?.focus() } }; window.addEventListener('keydown', close); return () => window.removeEventListener('keydown', close) }, [onClose])
  return <div className="modal-backdrop" onClick={onClose}><section className="help-dialog" role="dialog" aria-modal="true" aria-labelledby="help-title" onClick={e => e.stopPropagation()}><button ref={closeRef} className="icon-button modal-close" onClick={onClose} aria-label="Fermer l’aide"><X size={19} /></button><span className="help-icon"><Microscope size={30} /></span><span className="eyebrow">BIENVENUE SUR NEUROVIA</span><h2 id="help-title">Une lecture plus claire<br />des trajectoires.</h2><div className="help-steps"><p><b>01</b><span><strong>Explorez la cohorte</strong>Recherchez un identifiant PPMI ou utilisez les filtres.</span></p><p><b>02</b><span><strong>Choisissez votre modèle</strong>Comparez les prévisions à 12 et 24 mois et leurs dates index.</span></p><p><b>03</b><span><strong>Interprétez avec contexte</strong>Consultez les modalités, les visites et les limites des résultats.</span></p></div><p className="help-note">Cette interface explore une cohorte de recherche et des modèles existants. Les résultats ne constituent pas une recommandation thérapeutique.</p></section></div>
}

export default function App() {
  const [status, setStatus] = useState<EngineStatus>(initialStatus)
  const [connectionError, setConnectionError] = useState<string | null>(null)
  const [attempt, setAttempt] = useState(0)
  const [splash, setSplash] = useState(true)
  const [leaving, setLeaving] = useState(false)
  const [minimumElapsed, setMinimumElapsed] = useState(false)
  const [introInteracted, setIntroInteracted] = useState(false)
  const [enterRequested, setEnterRequested] = useState(false)
  const [page, setPage] = useState<Page>('dashboard')
  const [sidebar, setSidebar] = useState(false)
  const [help, setHelp] = useState(false)
  const [overview, setOverview] = useState<Overview | null>(null)
  const [patients, setPatients] = useState<Patient[]>([])
  const [selected, setSelected] = useState(3054)
  const [result, setResult] = useState<Prediction | null>(null)
  const [model, setModel] = useState<ModelId>('late_fusion')
  const [search, setSearch] = useState('')
  const [group, setGroup] = useState<Group>('all')
  const [dataError, setDataError] = useState('')
  const [predictionError, setPredictionError] = useState('')
  const [predictionBusy, setPredictionBusy] = useState(false)
  const [refresh, setRefresh] = useState(0)
  const [motion, setMotion] = useState(() => localStorage.getItem('neurovia-motion') !== 'reduced')

  useEffect(() => {
    if (!splash) return
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => { document.body.style.overflow = previousOverflow }
  }, [splash])
  useEffect(() => { document.documentElement.dataset.motion = motion ? 'full' : 'reduced'; localStorage.setItem('neurovia-motion', motion ? 'full' : 'reduced') }, [motion])
  useEffect(() => { const timer = setTimeout(() => setMinimumElapsed(true), motion && !window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 10000 : 500); return () => clearTimeout(timer) }, [attempt, motion])
  useEffect(() => {
    const controller = new AbortController()
    let timeout: ReturnType<typeof setTimeout>
    async function poll() {
      try {
        const value = await api<EngineStatus>('/status', { signal: controller.signal })
        setStatus(value); setConnectionError(value.error)
        if (value.state === 'error') return
        if (value.state !== 'ready') timeout = setTimeout(poll, 650)
      } catch (e) { if (!controller.signal.aborted) setConnectionError(e instanceof Error ? e.message : 'Connexion au serveur indisponible.') }
    }
    api<EngineStatus>('/initialize', { method: 'POST', signal: controller.signal }).then(value => { setStatus(value); void poll() }).catch(e => { if (!controller.signal.aborted) setConnectionError(`Impossible de joindre le moteur d’analyse. ${e.message}`) })
    return () => { controller.abort(); clearTimeout(timeout) }
  }, [attempt])
  useEffect(() => {
    if (status.state !== 'ready') return
    const controller = new AbortController()
    setDataError('')
    Promise.all([api<Overview>('/overview', { signal: controller.signal }), api<{ items: Patient[] }>('/patients', { signal: controller.signal })]).then(([data, list]) => {
      setOverview(data); setPatients(list.items)
    }).catch(e => { if (!controller.signal.aborted) setDataError(e.message) })
    return () => controller.abort()
  }, [status.state, refresh])
  useEffect(() => {
    if (status.state !== 'ready') return
    const controller = new AbortController()
    setPredictionBusy(true); setPredictionError('')
    api<Prediction>(`/patients/${selected}`, { signal: controller.signal }).then(setResult).catch(e => { if (!controller.signal.aborted) setPredictionError(e.message) }).finally(() => { if (!controller.signal.aborted) setPredictionBusy(false) })
    return () => controller.abort()
  }, [selected, status.state, refresh])
  useEffect(() => {
    if (!splash || !minimumElapsed || !overview || !result || status.state !== 'ready' || connectionError || dataError || predictionError || (introInteracted && !enterRequested)) return
    setLeaving(true)
    const timer = setTimeout(() => { setSplash(false); setLeaving(false) }, motion ? 850 : 50)
    return () => clearTimeout(timer)
  }, [minimumElapsed, overview, result, status.state, splash, motion, introInteracted, enterRequested, connectionError, dataError, predictionError])

  const filtered = useMemo(() => patients.filter(p => (!search || String(p.id).includes(search.trim())) && (group === 'all' || group === 'rapid' && p.rapid_progression || group === 'stable' && p.has_outcome && !p.rapid_progression || group === 'history' && p.visits >= 4 || group === 'imaging' && p.putamen_sbr != null && p.putamen_sbr < 1)), [patients, search, group])
  const selectedModel = models.find(m => m.id === model)!
  const pred = result?.predictions[model]
  const p12 = pred?.['12m'], p24 = pred?.['24m']
  const ready = status.state === 'ready'
  const patientMatches = result?.patient.id === selected
  function navigate(value: Page) { setPage(value); setSidebar(false); window.scrollTo({ top: 0, behavior: 'instant' }) }
  function selectPatient(id: number) { setSelected(id); navigate('dashboard') }
  function retry() { setConnectionError(null); setDataError(''); setPredictionError(''); setAttempt(a => a + 1); setRefresh(r => r + 1) }
  function enterIntro() { setEnterRequested(true); setMinimumElapsed(true) }
  function replay() { setMinimumElapsed(false); setIntroInteracted(false); setEnterRequested(false); setLeaving(false); setSplash(true); setAttempt(a => a + 1) }
  const titles: Record<Page, string> = { dashboard: 'Tableau de bord', patients: 'Explorer les patients', models: 'Modèles & performances', settings: 'Votre espace de travail' }

  return <>
    {splash && <CinematicIntro status={status} error={connectionError || dataError || predictionError || null} leaving={leaving} motionEnabled={motion} onRetry={retry} onContinue={enterIntro} onInteraction={() => setIntroInteracted(true)} />}
    <div className={`app-shell ${splash ? 'app-behind-intro' : 'app-visible'}`} inert={splash || help ? true : undefined}>
      {sidebar && <button className="sidebar-backdrop" aria-label="Fermer la navigation" onClick={() => setSidebar(false)} />}
      <aside className={`sidebar ${sidebar ? 'sidebar-open' : ''}`}><div className="sidebar-brand"><Brand /><button className="icon-button sidebar-close" onClick={() => setSidebar(false)} aria-label="Fermer le menu"><PanelLeftClose size={18} /></button></div><div className="workspace-badge"><span><FlaskConical size={14} /> Espace de recherche</span><ChevronDown size={13} /></div><span className="nav-label">ESPACE CLINIQUE</span><nav>{([
        ['dashboard', LayoutDashboard, 'Tableau de bord'], ['patients', Users, 'Patients'], ['models', Cpu, 'Modèles & analyses'],
      ] as const).map(([key, Icon, label]) => <button key={key} className={page === key ? 'nav-item nav-active' : 'nav-item'} onClick={() => navigate(key)}><Icon size={18} strokeWidth={1.7} /><span>{label}</span>{key === 'patients' && overview && <small>{overview.patients}</small>}{page === key && <span className="nav-active-dot" />}</button>)}</nav><div className="sidebar-divider" /><span className="nav-label">PRÉFÉRENCES</span><button className={page === 'settings' ? 'nav-item nav-active' : 'nav-item'} onClick={() => navigate('settings')}><Settings2 size={18} strokeWidth={1.7} /><span>Paramètres</span></button><button className="nav-item" onClick={() => setHelp(true)}><CircleHelp size={18} strokeWidth={1.7} /><span>Guide de prise en main</span></button><div className="sidebar-bottom"><div className="research-note"><span className="research-icon"><Microscope size={20} /></span><strong>La recherche, connectée.</strong><p>Trois modèles. Une vision<br />plus complète des trajectoires.</p><button onClick={() => navigate('models')}>Explorer les modèles <ArrowUpRight size={14} /></button></div><div className="local-status"><span className="pulse-dot" /><span>Moteur local opérationnel</span><ShieldCheck size={14} /></div><button className="workspace-user" onClick={() => navigate('settings')}><span className="user-avatar"><Stethoscope size={18} /></span><span><strong>Espace de recherche</strong><small>Cohorte PPMI · accès local</small></span><Settings2 size={15} /></button></div></aside>
      <div className="main-shell"><header className="topbar"><div className="breadcrumb"><button className="icon-button mobile-menu" aria-label="Ouvrir le menu" onClick={() => setSidebar(true)}><Menu size={21} /></button><span>Espace clinique</span><ChevronRight size={13} /><strong>{titles[page]}</strong></div><div className="topbar-right"><span className="connection"><span className="tiny-dot" />{ready ? 'Modèles connectés' : 'Connexion…'}</span><span className="topbar-divider" /><button className="icon-button" title="Aide et contexte scientifique" aria-label="Aide et contexte scientifique" onClick={() => setHelp(true)}><CircleHelp size={18} /></button><span className="profile-dot">NV</span></div></header>
        <main className="main-content"><div className="page-heading"><div><div className="page-kicker"><span className="eyebrow">INTELLIGENCE MULTIMODALE</span><span className="kicker-line" /></div><h1>{titles[page]}<span className="heading-dot">.</span></h1><p>{page === 'dashboard' ? 'Chaque donnée raconte une partie de l’histoire. Relions-les.' : page === 'patients' ? 'Retrouvez les profils et les historiques de la cohorte de test.' : page === 'models' ? 'Trois approches complémentaires pour explorer la progression.' : 'Une expérience locale, pensée pour votre recherche.'}</p></div><div className="heading-actions">{page === 'dashboard' && <><button className="secondary-button" onClick={() => navigate('patients')}><Users size={15} /> Voir la cohorte</button><a className={`primary-button ${predictionBusy || !patientMatches ? 'disabled-link' : ''}`} href={`/api/patients/${selected}/export`} download><Download size={15} /> Exporter le patient</a></>}{page === 'patients' && <Pill tone="purple"><Users size={13} /> {integer(filtered.length)} patients</Pill>}</div></div>
          {dataError && <div className="error-banner" role="alert">{dataError}<button onClick={retry}>Réessayer</button></div>}
          {page === 'dashboard' && <>
            <section className="research-banner"><div className="banner-icon"><FlaskConical size={21} strokeWidth={1.6} /></div><div><strong>La recherche prend une nouvelle dimension.</strong><p>Cohorte de test PPMI · Données longitudinales réelles · Prévisions à 12 et 24 mois</p><div className="banner-tags"><span><CheckCheck size={12} /> Données locales</span><span><CheckCheck size={12} /> Modèles préentraînés</span><span><CheckCheck size={12} /> Exploration de recherche</span></div></div><button className="text-button" onClick={() => setHelp(true)}>Découvrir l’espace <ArrowRight size={15} /></button><div className="banner-orb" /></section>
            <div className="patient-toolbar"><div className="selected-patient"><span className="patient-avatar avatar-1">{String(selected).slice(-2)}</span><label><small>PROFIL PATIENT</small><span className="select-wrap"><select aria-label="Patient sélectionné" value={selected} onChange={e => setSelected(Number(e.target.value))}>{patients.map(p => <option value={p.id} key={p.id}>Patient #{p.id}</option>)}</select><ChevronDown size={13} /></span></label>{result && patientMatches && <span className="patient-context">{fmt(result.patient.age, 0)} ans <i /> {result.patient.sex} <i /> Stade H&Y {fmt(result.patient.stage, 0)}</span>}</div><label className="model-picker"><span><Cpu size={15} /> Modèle actif</span><span className="select-wrap"><select aria-label="Modèle actif" value={model} onChange={e => setModel(e.target.value as ModelId)}>{models.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}</select><ChevronDown size={13} /></span></label></div>
            {predictionError && <div className="error-banner" role="alert">{predictionError}<button onClick={() => setRefresh(r => r + 1)}>Réessayer</button></div>}
            {predictionBusy || !patientMatches ? <div className="prediction-loading" role="status"><LoaderCircle size={25} className="spinning" /><strong>Analyse de la trajectoire du patient #{selected}</strong><span>Les modèles croisent les données disponibles…</span></div> : result && <>
              <div className="metrics-grid"><Metric label="Score moteur actuel" value={fmt(result.patient.score)} unit="/ 132" icon={Activity} tone="purple" detail={`Visite du ${date(result.patient.last_visit)}`} spark={result.history.flatMap(v => v.score == null ? [] : [v.score])} /><Metric label="Prévision à 12 mois" value={fmt(p12?.score)} unit="pts" icon={TrendingUp} tone="teal" detail={p12 ? `${delta(p12.delta)} depuis la visite index` : 'Visite index non disponible'} spark={p12 ? [p12.index_score, p12.score] : []} /><Metric label="Prévision à 24 mois" value={fmt(p24?.score)} unit="pts" icon={ChartNoAxesCombined} tone="blue" detail={p24 ? `${delta(p24.delta)} depuis la visite index` : 'Visite index non disponible'} spark={p24 ? [p24.index_score, p24.score] : []} /><Metric label="Historique clinique" value={String(result.history.length)} unit="visites" icon={History} tone="peach" detail={`${date(result.history[0].date)} — ${date(result.patient.last_visit)}`}  /></div>
              <div className="analysis-grid"><Trajectory result={result} model={model} /><Attribution result={result} model={model} /></div>
              <div className="insights-grid"><Card className="biomarkers-card"><div className="card-header"><div><h3>Le profil, en détail</h3><p>Sous-domaines moteurs · dernière visite</p></div><span className="soft-icon"><SlidersHorizontal size={18} /></span></div><div className="biomarkers-list">{[['np3_tremor', 'Tremblement', 40], ['np3_bradykinesia', 'Bradykinésie', 44], ['np3_rigidity', 'Rigidité', 20], ['np3_axial', 'Marche & axial', 28]].map(([key, label, max], i) => <div className="biomarker" key={key}><div><span><i className={`biomarker-dot bio-${i}`} />{label}</span><strong>{fmt(result.biomarkers[key])}<small>pts</small></strong></div><div className="biomarker-track"><i className={`bio-bg-${i}`} style={{ width: `${Math.min(100, (result.biomarkers[key] ?? 0) / Number(max) * 100)}%` }} /></div></div>)}</div></Card><Card className="summary-card"><div className="card-header"><div><h3>Lecture du modèle</h3><p>{selectedModel.name} · patient #{selected}</p></div><span className="soft-icon"><AudioLines size={18} /></span></div><div className="prediction-summary"><span className={`summary-icon ${pred?.probability != null && pred.probability >= .5 ? 'summary-amber' : ''}`}>{pred?.probability == null ? <ChartNoAxesCombined size={23} /> : <Activity size={23} />}</span><div><strong>{pred?.probability == null ? 'Prévisions par horizon' : `Probabilité de progression : ${fmt(pred.probability * 100)} %`}</strong><p>{pred?.probability == null ? 'Ce modèle ne fournit pas de probabilité de classification dans cette interface.' : 'Sortie du classifieur pour une hausse du score moteur d’au moins 3,5 points à 12 mois.'}</p></div></div><div className="summary-detail"><span><span className="tiny-dot" /> Moteur d’inférence local</span><span>12 m · {p12 ? date(p12.index_date) : 'N/D'}<br />24 m · {p24 ? date(p24.index_date) : 'N/D'}</span></div></Card><Card className="neural-card"><img src="/media/neural-poster.jpg" alt="Cerveau et réseau neuronal, visuel artistique de l’introduction" /><div className="neural-card-overlay" /><div className="neural-card-content"><span className="eyebrow">LES LIENS FONT LE SENS</span><h3>Une vision<br />multimodale.</h3><button onClick={replay} aria-label="Revoir la vidéo d’introduction"><CirclePlay size={17} /> Revoir l’introduction <ArrowUpRight size={14} /></button></div></Card></div>
            </>}
            <Card className="cohort-card"><div className="card-header"><div><h3>Explorer la cohorte</h3><p>{integer(overview?.patients ?? 0)} patients · cohorte de test PPMI</p></div><button className="text-button" onClick={() => navigate('patients')}>Tous les patients <ArrowRight size={15} /></button></div><PatientTable patients={patients} selected={selected} onSelect={selectPatient} compact /></Card>
          </>}
          {page === 'patients' && <><div className="cohort-summary"><Card><span className="soft-icon"><Users size={21} /></span><div><strong>{integer(overview?.patients ?? 0)}</strong><span>Patients de la cohorte</span></div></Card><Card><span className="soft-icon teal-text"><History size={21} /></span><div><strong>{integer(overview?.visits ?? 0)}</strong><span>Visites longitudinales</span></div></Card><Card><span className="soft-icon peach-text"><Activity size={21} /></span><div><strong>{fmt(overview?.mean_score)}</strong><span>Score moteur moyen · dernière visite</span></div></Card></div><Card className="cohort-card"><div className="patient-filters"><label className="search-field"><Search size={17} /><input value={search} onChange={e => setSearch(e.target.value)} placeholder="Rechercher un identifiant patient…" aria-label="Rechercher un patient" />{search && <button className="icon-button" aria-label="Effacer la recherche" onClick={() => setSearch('')}><X size={14} /></button>}</label><label className="filter-select"><SlidersHorizontal size={16} /><select value={group} onChange={e => setGroup(e.target.value as Group)} aria-label="Filtrer les patients">{groups.map(g => <option value={g.id} key={g.id}>{g.label}</option>)}</select><ChevronDown size={13} /></label></div><PatientTable patients={filtered} selected={selected} onSelect={selectPatient} /></Card></>}
          {page === 'models' && <ModelPage active={model} onChoose={value => { setModel(value); navigate('dashboard') }} />}
          {page === 'settings' && <div className="settings-grid"><Card><SectionTitle eyebrow="EXPÉRIENCE" title="À votre rythme" /><div className="setting-row"><div><strong>Animations & transitions</strong><p>Adoucir le passage entre l’introduction et votre espace.</p></div><button className={`switch ${motion ? 'switch-on' : ''}`} role="switch" aria-checked={motion} aria-label="Animations et transitions" onClick={() => setMotion(!motion)}><span /></button></div><div className="setting-row"><div><strong>La connexion, en images</strong><p>Revoir la vidéo neuronale de l’introduction.</p></div><button className="secondary-button" onClick={replay}><CirclePlay size={16} /> Revoir</button></div><div className="setting-row"><div><strong>Comprendre l’interface</strong><p>Les premiers repères pour explorer les trajectoires.</p></div><button className="secondary-button" onClick={() => setHelp(true)}><CircleHelp size={16} /> Ouvrir le guide</button></div></Card><Card><SectionTitle eyebrow="MOTEUR LOCAL" title="Votre connexion" /><div className="system-ready"><ShieldCheck size={25} /><div><strong>Les trois modèles sont disponibles</strong><p>Les données et les prédictions restent sur votre machine.</p></div></div>{models.map(m => <div className="system-model" key={m.id}><m.icon size={17} /><span>{m.name}</span><Pill tone="teal"><Check size={12} /> Prêt</Pill></div>)}<div className="system-meta"><span>Interface</span><strong>React · TypeScript</strong><span>Moteur</span><strong>FastAPI · Python</strong><span>Données</span><strong>PPMI · {overview?.patients} patients</strong></div></Card></div>}
          <footer className="page-footer"><div><Brand /><span>La science des trajectoires humaines.</span></div><p><FlaskConical size={13} /> Exploration de recherche · cohorte historique · validation indépendante requise.</p><span>PPMI DATA · LOCAL</span></footer>
        </main>
      </div>
    </div>
    {help && <HelpDialog onClose={() => setHelp(false)} />}
  </>
}
