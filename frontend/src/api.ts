export async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`/api${path}`, { ...options, signal: options.signal ?? AbortSignal.timeout(20000) })
  if (!response.ok) {
    let message = `Le serveur a répondu avec le code ${response.status}.`
    try {
      const body = await response.json()
      if (typeof body.detail === 'string') message = body.detail
      else if (body.detail?.message) message = body.detail.message
    } catch { /* Keep the HTTP error if the response isn't JSON. */ }
    throw new Error(message)
  }
  return response.json() as Promise<T>
}
