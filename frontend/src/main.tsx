import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './styles.css'
import './readability.css'
import './intro.css'

class ErrorBoundary extends React.Component<React.PropsWithChildren, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() { return { failed: true } }
  render() {
    if (this.state.failed) return <main className="fatal-error"><h1>Un instant, quelque chose a interrompu l’affichage.</h1><p>Vos données sont conservées. Rechargez l’interface pour reprendre.</p><button className="primary-button" onClick={() => location.reload()}>Recharger l’interface</button></main>
    return this.props.children
  }
}

ReactDOM.createRoot(document.getElementById('root')!).render(<ErrorBoundary><App /></ErrorBoundary>)
