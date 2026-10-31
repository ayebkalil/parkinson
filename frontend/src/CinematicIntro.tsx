import { useEffect, useRef, useState } from 'react'
import {
  ArrowRight, BrainCircuit, Check, CheckCheck, Expand, LoaderCircle,
  Pause, Play, RotateCcw, ShieldCheck,
} from 'lucide-react'
import type { EngineStatus } from './types'

interface IntroProps {
  status: EngineStatus
  error: string | null
  leaving: boolean
  motionEnabled: boolean
  onRetry: () => void
  onContinue: () => void
  onInteraction: () => void
}

const stages = [
  { label: 'Cohorte PPMI', at: 0, done: 32 },
  { label: 'Prétraitement', at: 32, done: 48 },
  { label: 'Modèles IA', at: 48, done: 94 },
  { label: 'Votre espace', at: 94, done: 100 },
]

export default function CinematicIntro({ status, error, leaving, motionEnabled, onRetry, onContinue, onInteraction }: IntroProps) {
  const video = useRef<HTMLVideoElement>(null)
  const scene = useRef<HTMLElement>(null)
  const [playing, setPlaying] = useState(false)
  const [hasFrame, setHasFrame] = useState(false)
  const [videoFailed, setVideoFailed] = useState(false)
  const [manualMotion, setManualMotion] = useState(false)
  const [systemReduced, setSystemReduced] = useState(() => window.matchMedia('(prefers-reduced-motion: reduce)').matches)
  const [duration, setDuration] = useState(0)
  const [fullscreen, setFullscreen] = useState(false)
  const [mediaMessage, setMediaMessage] = useState('')
  const reduced = !motionEnabled || systemReduced
  const ready = status.state === 'ready' && !error
  const progress = Math.max(0, Math.min(100, Math.round(status.progress)))

  useEffect(() => {
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)')
    const update = () => setSystemReduced(preference.matches)
    preference.addEventListener('change', update)
    return () => preference.removeEventListener('change', update)
  }, [])

  useEffect(() => {
    const update = () => setFullscreen(document.fullscreenElement === scene.current)
    document.addEventListener('fullscreenchange', update)
    return () => document.removeEventListener('fullscreenchange', update)
  }, [])

  useEffect(() => {
    if (reduced && !manualMotion) video.current?.pause()
  }, [reduced, manualMotion])

  async function togglePlayback() {
    onInteraction()
    setMediaMessage('')
    const element = video.current
    if (!element || videoFailed) return
    if (element.paused) {
      setManualMotion(true)
      try { await element.play() }
      catch { setMediaMessage('La lecture n’a pas pu démarrer. Vous pouvez accéder à votre espace dès qu’il est prêt.') }
    } else element.pause()
  }

  async function toggleFullscreen() {
    onInteraction()
    try {
      if (document.fullscreenElement === scene.current) await document.exitFullscreen()
      else await scene.current?.requestFullscreen()
    } catch { setMediaMessage('Le navigateur ne permet pas le plein écran dans cette fenêtre.') }
  }

  async function enterWorkspace() {
    // A fullscreen element disappears when its intro unmounts; close it first
    // so the subsequent dashboard keeps the user's normal browser viewport.
    if (document.fullscreenElement === scene.current) {
      try { await document.exitFullscreen() } catch { /* The browser also closes fullscreen on unmount. */ }
    }
    onContinue()
  }

  return <section ref={scene} className={`intro intro-cinema ${leaving ? 'intro-leaving' : ''} ${manualMotion ? 'cinema-allow-motion' : ''}`} aria-label="Introduction de Neurovia et préparation des modèles">
    <div className="cinema-ambient" aria-hidden="true" />
    <div className="cinema-stage" aria-hidden="true">
      <img className={`intro-poster cinema-film ${hasFrame && !videoFailed && (!reduced || manualMotion) ? 'poster-hidden' : ''}`} src="/media/neural-poster.jpg" alt="" />
      {!videoFailed && <video ref={video} className="cinema-film cinema-video" autoPlay={!reduced} muted loop playsInline preload="auto" poster="/media/video-poster.jpg"
        onPlaying={() => { setPlaying(true); setHasFrame(true) }}
        onPause={() => setPlaying(false)}
        onLoadedMetadata={event => setDuration(event.currentTarget.duration)}
        onError={() => { setVideoFailed(true); setPlaying(false) }}>
        <source src="/media/neural-intro.mp4" type="video/mp4" />
      </video>}
    </div>
    <div className="cinema-shade" aria-hidden="true" />
    <div className="cinema-grain" aria-hidden="true" />
    <div className="cinema-halo cinema-halo-one" aria-hidden="true" />
    <div className="cinema-halo cinema-halo-two" aria-hidden="true" />

    <header className="cinema-header">
      <div className="brand brand-light"><span className="brand-mark"><BrainCircuit size={25} strokeWidth={1.6} /></span><div><strong>neurovia<span>.</span></strong><small>INTELLIGENCE PARKINSON</small></div></div>
      <div className={`cinema-engine-state ${ready ? 'engine-is-ready' : ''}`}><span className="cinema-status-dot" /><span>{error ? 'Connexion interrompue' : ready ? 'Votre espace est prêt' : 'Préparation de votre espace'}</span></div>
    </header>

    <div className="cinema-layout">
      <div className="cinema-copy">
        <div className="cinema-eyebrow"><span /> LA SCIENCE DES TRAJECTOIRES HUMAINES</div>
        <h1><span>Comprendre.</span><em>Anticiper.</em></h1>
        <p>La clinique, l’imagerie et l’intelligence artificielle.<br />Une nouvelle perspective sur la progression de Parkinson.</p>
        <div className="cinema-disciplines" aria-label="Les domaines réunis dans Neurovia"><span>Clinique</span><i /><span>Imagerie</span><i /><span>Trajectoires</span></div>
      </div>

      <div className={`cinema-loading-dock ${ready ? 'dock-is-ready' : ''} ${error ? 'dock-has-error' : ''}`}>
        <div className="cinema-load-heading">
          <span className="cinema-load-symbol">{ready ? <CheckCheck size={20} /> : error ? <RotateCcw size={18} /> : <LoaderCircle size={19} className="spinning" />}</span>
          <div role="status" aria-live="polite"><strong>{error ? 'Reprenons la connexion' : ready ? 'Les modèles sont initialisés' : 'Nous relions vos données'}</strong><span>{error ? 'Votre espace reste disponible après un nouvel essai.' : status.stage}</span></div>
          <span className="cinema-percentage">{progress}<small>%</small></span>
        </div>
        <div className="cinema-load-track" role="progressbar" aria-valuenow={progress} aria-valuemin={0} aria-valuemax={100} aria-label="Progression réelle du chargement des modèles"><span style={{ width: `${progress}%` }} /></div>
        <div className="cinema-load-stages">{stages.map((stage, index) => <div key={stage.label} className={progress >= stage.done ? 'is-complete' : progress >= stage.at ? 'is-current' : ''}>
          <span>{progress >= stage.done ? <Check size={10} /> : `0${index + 1}`}</span>{stage.label}
        </div>)}</div>
        {error ? <div className="cinema-error" role="alert"><p>{error}</p><button onClick={onRetry}><RotateCcw size={14} /> Réessayer</button></div> : <div className="cinema-dock-footer">
          <span><ShieldCheck size={13} /> {status.loaded_models.length} / 3 moteurs chargés</span>
          <button className="cinema-enter" disabled={!ready || leaving} onClick={() => void enterWorkspace()}>{ready ? 'Ouvrir mon espace' : 'Initialisation en cours'}<ArrowRight size={16} /></button>
        </div>}
      </div>
    </div>

    <aside className="cinema-visual-caption" aria-label="Contexte du visuel"><span>EXPLORATION NEURONALE</span><p>Le mouvement des connexions.<br />La force des perspectives.</p><small>Visualisation artistique</small></aside>
    <footer className="cinema-footer">
      <span className="cinema-local"><ShieldCheck size={14} /> Modèles et données chargés localement</span>
      <div className="cinema-playback"><span className={`cinema-playback-state ${playing ? 'is-playing' : ''}`}><i />{videoFailed ? 'Image de remplacement' : playing ? 'Lecture en boucle' : 'Lecture en pause'}{duration > 0 && <small>{Math.round(duration)} s</small>}</span>
        <div className="cinema-media-buttons">
          <button disabled={videoFailed || leaving} aria-label={playing ? 'Mettre la vidéo en pause' : 'Lire la vidéo'} title={playing ? 'Mettre en pause' : 'Lire la vidéo'} onClick={() => void togglePlayback()}>{playing ? <Pause size={15} /> : <Play size={15} />}</button>
          <button disabled={leaving} aria-label={fullscreen ? 'Quitter le plein écran' : 'Voir l’introduction en plein écran'} title={fullscreen ? 'Quitter le plein écran' : 'Plein écran'} onClick={() => void toggleFullscreen()}><Expand size={15} /></button>
        </div>
      </div>
    </footer>
    {mediaMessage && <p className="cinema-media-message" role="status">{mediaMessage}</p>}
  </section>
}
