const envBase = import.meta.env.VITE_API_BASE

// DEV falls back to the local backend. PROD does NOT silently fall back to
// localhost:8000 (which meant every visitor hit their own machine) — the
// base must come from VITE_API_BASE at build time. Relative URLs against a
// static-frontend origin cannot work either (same-origin /api/* does not
// exist there), so a missing base is treated as a configuration error on
// every request instead of mysterious network/404 failures.
const API_BASE = envBase || (import.meta.env.DEV ? 'http://localhost:8000' : '')
const PROD_CONFIG_MISSING = Boolean(import.meta.env.PROD && !envBase)

if (PROD_CONFIG_MISSING) {
  // eslint-disable-next-line no-console
  console.error(
    '[api] VITE_API_BASE is not set at build time. ' +
      'Set it in the frontend service environment (e.g. the backend Render URL) and redeploy.',
  )
}

export { API_BASE }

// Default request timeout. Detection (model warm start ~8-9 s, warm ~0.5-1.2 s
// locally on free-CPU hardware) plus Render Free cold-start wake-up needs more
// headroom than a typical API call.
const DEFAULT_TIMEOUT_MS = 45000

// Detection is the heaviest call: on Render Free the first request after a
// cold start pays instance wake-up AND full model load (~9 s measured locally
// cold), so it gets its own generous ceiling.
const DETECTION_TIMEOUT_MS = 90000

// Video upload acceptance is quick (processing itself is a polled background
// job); the larger 50 MB upload body just needs upload headroom.
const VIDEO_SUBMIT_TIMEOUT_MS = 60000

// Status polling is light and frequent; a short ceiling keeps the poller
// from stacking up if the backend stalls.
const POLL_TIMEOUT_MS = 10000

class ApiError extends Error {
  constructor(kind, message, status) {
    super(message)
    this.name = 'ApiError'
    this.kind = kind // 'network' | 'timeout' | 'http' | 'config'
    this.status = status
  }
}

async function request(path, options = {}) {
  const { timeoutMs = DEFAULT_TIMEOUT_MS, ...fetchOptions } = options

  if (PROD_CONFIG_MISSING) {
    throw new ApiError(
      'config',
      'Frontend misconfigured: VITE_API_BASE was not set at build time, so the API address is unknown.',
      undefined,
    )
  }

  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)

  let res
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...fetchOptions,
      signal: controller.signal,
    })
  } catch (err) {
    if (controller.signal.aborted) {
      throw new ApiError('timeout', 'The server took too long to respond. Please try again.', undefined)
    }
    throw new ApiError('network', 'Cannot reach the server. Check your connection or try again later.', undefined)
  } finally {
    clearTimeout(timer)
  }

  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail || body.error || detail
    } catch {
      // response wasn't JSON, keep statusText
    }
    const err = new ApiError('http', detail, res.status)
    throw err
  }

  return res.json()
}

export const api = {
  health: () => request('/api/health', { timeoutMs: 15000 }),
  readiness: () => request('/api/readiness', { timeoutMs: 15000 }),
  dashboard: () => request('/api/dashboard'),
  batchResults: (params = {}) =>
    request(`/api/batch/results?${new URLSearchParams(params)}`),
  batchAnalytics: () => request('/api/batch/analytics'),
  videoSummary: () => request('/api/video/summary'),
  videoDetections: (params = {}) =>
    request(`/api/video/detections?${new URLSearchParams(params)}`),
  plates: (params = {}) => request(`/api/plates?${new URLSearchParams(params)}`),
  plateDetail: (plate) =>
    request(`/api/plates/${encodeURIComponent(plate)}`, { timeoutMs: 15000 }),
  // RESTORED: System.jsx (system), Detection.jsx + Overview.jsx (history) and
  // ModelPerformance.jsx (model) still call these — the previous edit removed
  // them and broke those pages with a TypeError at render time.
  model: () => request('/api/model'),
  system: () => request('/api/system'),
  history: (params = {}) => request(`/api/history?${new URLSearchParams(params)}`),
  detectImage: (file, conf = 0.25) => {
    const form = new FormData()
    form.append('file', file)
    return request(`/api/detect/image?conf=${conf}`, {
      method: 'POST',
      body: form,
      timeoutMs: DETECTION_TIMEOUT_MS,
    })
  },
  processVideo: (file, frameSkip = 15) => {
    const form = new FormData()
    form.append('file', file)
    return request(`/api/process/video?frame_skip=${frameSkip}`, {
      method: 'POST',
      body: form,
      timeoutMs: VIDEO_SUBMIT_TIMEOUT_MS,
    })
  },
  processVideoStatus: (jobId) =>
    request(`/api/process/video/${jobId}`, { timeoutMs: POLL_TIMEOUT_MS }),
}
