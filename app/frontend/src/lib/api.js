const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'
export { API_BASE }

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, options)
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail || body.error || detail
    } catch {
      // response wasn't JSON, keep statusText
    }
    const err = new Error(detail)
    err.status = res.status
    throw err
  }
  return res.json()
}

export const api = {
  health: () => request('/api/health'),
  dashboard: () => request('/api/dashboard'),
  batchResults: (params = {}) =>
    request(`/api/batch/results?${new URLSearchParams(params)}`),
  batchAnalytics: () => request('/api/batch/analytics'),
  videoSummary: () => request('/api/video/summary'),
  videoDetections: (params = {}) =>
    request(`/api/video/detections?${new URLSearchParams(params)}`),
  plates: (params = {}) => request(`/api/plates?${new URLSearchParams(params)}`),
  plateDetail: (plate) => request(`/api/plates/${encodeURIComponent(plate)}`),
  model: () => request('/api/model'),
  system: () => request('/api/system'),
  history: (params = {}) => request(`/api/history?${new URLSearchParams(params)}`),
  detectImage: (file, conf = 0.25) => {
    const form = new FormData()
    form.append('file', file)
    return request(`/api/detect/image?conf=${conf}`, { method: 'POST', body: form })
  },
  processVideo: (file, frameSkip = 15) => {
    const form = new FormData()
    form.append('file', file)
    return request(`/api/process/video?frame_skip=${frameSkip}`, {
      method: 'POST',
      body: form,
    })
  },
  processVideoStatus: (jobId) => request(`/api/process/video/${jobId}`),
}
