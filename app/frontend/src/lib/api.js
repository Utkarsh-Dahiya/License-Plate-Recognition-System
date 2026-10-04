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

export { API_BASE, ApiError }

// Image uploads: the backend accepts any `image/*` content type and caps the
// body at MAX_IMAGE_BYTES (10 MB) in app/backend/main.py. Mirroring both here
// catches a bad file instantly instead of after a slow upload.
export const ACCEPTED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp']
export const ACCEPTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp']
export const MAX_IMAGE_BYTES = 10 * 1024 * 1024

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
    this.kind = kind // 'network' | 'timeout' | 'aborted' | 'http' | 'config' | 'malformed'
    this.status = status
  }
}

/**
 * FastAPI returns `detail` as a string for HTTPException and as an ARRAY of
 * {loc, msg, type} objects for request-validation failures (422). Passing the
 * raw array to the UI produced "[object Object]"; flatten it instead.
 */
function extractDetail(body, fallback) {
  const raw = body && (body.detail !== undefined ? body.detail : body.error)
  if (typeof raw === 'string' && raw.trim()) return raw.trim()
  if (Array.isArray(raw)) {
    const parts = raw
      .map((item) => {
        if (typeof item === 'string') return item
        if (item && typeof item.msg === 'string') {
          const field = Array.isArray(item.loc) ? item.loc[item.loc.length - 1] : null
          return field ? `${field}: ${item.msg}` : item.msg
        }
        return null
      })
      .filter(Boolean)
    if (parts.length) return parts.join(' ')
  }
  if (raw && typeof raw === 'object' && typeof raw.msg === 'string') return raw.msg
  return fallback
}

/**
 * Shared fetch wrapper.
 *
 * `options.signal` lets callers cancel in-flight work (a reset, a new sample,
 * an unmount). The caller-controlled abort is reported as its own `kind` so
 * it is never mistaken for a timeout or a network failure.
 */
async function rawRequest(path, options = {}) {
  const { timeoutMs = DEFAULT_TIMEOUT_MS, signal, ...fetchOptions } = options

  if (PROD_CONFIG_MISSING) {
    throw new ApiError(
      'config',
      'Frontend misconfigured: VITE_API_BASE was not set at build time, so the API address is unknown.',
      undefined,
    )
  }

  const controller = new AbortController()
  let timedOut = false
  const timer = setTimeout(() => {
    timedOut = true
    controller.abort()
  }, timeoutMs)

  const onExternalAbort = () => controller.abort()
  if (signal) {
    if (signal.aborted) {
      clearTimeout(timer)
      throw new ApiError('aborted', 'Request cancelled.', undefined)
    }
    signal.addEventListener('abort', onExternalAbort)
  }

  let res
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...fetchOptions,
      signal: controller.signal,
    })
  } catch (err) {
    if (signal?.aborted) {
      throw new ApiError('aborted', 'Request cancelled.', undefined)
    }
    if (timedOut) {
      throw new ApiError('timeout', 'The server took too long to respond. Please try again.', undefined)
    }
    throw new ApiError('network', 'Cannot reach the server. Check your connection or try again later.', undefined)
  } finally {
    clearTimeout(timer)
    if (signal) signal.removeEventListener('abort', onExternalAbort)
  }

  if (!res.ok) {
    let detail = res.statusText || `Request failed with status ${res.status}`
    try {
      const body = await res.json()
      detail = extractDetail(body, detail)
    } catch {
      // Non-JSON error body (proxy HTML, gateway page): keep the status text.
    }
    throw new ApiError('http', detail, res.status)
  }

  return res
}

async function request(path, options = {}) {
  const res = await rawRequest(path, options)

  // A 2xx that is not JSON is a broken/mismatched deployment, not an empty
  // result. Surfacing it as such avoids a confusing downstream crash.
  let payload
  try {
    payload = await res.json()
  } catch {
    throw new ApiError('malformed', 'The server returned a response that is not valid JSON.', res.status)
  }
  return payload
}

/** Blob variant used for binary sample downloads. */
async function requestBlob(path, options = {}) {
  const res = await rawRequest(path, options)
  try {
    return await res.blob()
  } catch {
    throw new ApiError('malformed', 'The server returned an unreadable image.', res.status)
  }
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
  historyStats: () => request('/api/history/stats'),
  // Export is a browser download (Content-Disposition attachment), so it
  // is a plain URL for an <a download> rather than a JSON request.
  historyExportUrl: (params = {}) =>
    `${API_BASE}/api/history/export?${new URLSearchParams(params)}`,
  // `conf` is OPTIONAL and is omitted from the query string when not
  // supplied, so the backend's own tuned default applies.
  //
  // This used to default to a hardcoded 0.25 and always send it, which
  // silently OVERRODE the backend default on every request. The backend
  // threshold is tuned against the labeled validation split; keeping one
  // source of truth means callers that do not care cannot drift from it.
  detectImage: (file, conf, { signal } = {}) => {
    const form = new FormData()
    form.append('file', file)
    const query = conf === undefined || conf === null ? '' : `?conf=${conf}`
    return request(`/api/detect/image${query}`, {
      method: 'POST',
      body: form,
      timeoutMs: DETECTION_TIMEOUT_MS,
      signal,
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
  samples: (options = {}) => request('/api/samples', options),
  sampleDetail: (id, options = {}) => request(`/api/samples/${id}`, options),
  sampleImageUrl: (id) => `${API_BASE}/api/samples/${id}/image`,
  // Uses the shared wrapper so sample loading gets the same timeout,
  // abort and error taxonomy as every other call (it used to be a bare
  // fetch with no timeout and an untyped Error).
  fetchSampleBlob: (id, options = {}) => requestBlob(`/api/samples/${id}/image`, options),
}
