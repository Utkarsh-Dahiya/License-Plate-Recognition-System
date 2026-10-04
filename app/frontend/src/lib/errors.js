/**
 * User-facing error translation.
 *
 * The UI never renders a raw exception, a traceback or a raw status text.
 * Everything that reaches a component goes through `toUserError()` first,
 * which returns a small, predictable shape:
 *
 *   { title, message, retryable, kind, status }
 *
 * `kind` mirrors the low-level cause so callers can branch (e.g. only offer
 * "Retry" when the failure is plausibly transient) without string matching.
 */

const TRUNCATE_AT = 240

/** Build a user-facing error directly (client-side validation, no network). */
export function userError({ title, message, retryable = false, kind = 'client' }) {
  return { title, message, retryable, kind, status: null, userFacing: true }
}

/**
 * Backend 500s can echo an exception repr. Keep the user-facing copy clean:
 * anything that smells like a traceback or is absurdly long is dropped in
 * favour of a generic explanation.
 */
export function sanitizeDetail(detail) {
  if (typeof detail !== 'string') return null
  const trimmed = detail.trim()
  if (!trimmed) return null
  if (trimmed.includes('Traceback (most recent call last)')) return null
  if (trimmed.includes('File "') && trimmed.includes('line ')) return null
  if (trimmed.length > TRUNCATE_AT) return `${trimmed.slice(0, TRUNCATE_AT).trimEnd()}…`
  return trimmed
}

const HTTP_COPY = {
  400: {
    title: 'That image could not be read',
    message:
      'The server rejected the uploaded file as a readable image. It may be corrupt, empty, or not an image at all.',
  },
  404: {
    title: 'Endpoint not found',
    message:
      'The backend did not recognise the requested endpoint. The API build may be out of date with this frontend.',
  },
  405: {
    title: 'Request not allowed',
    message: 'The backend rejected this request type. Reload the app and try again.',
  },
  413: {
    title: 'Image is too large',
    message: 'The uploaded image exceeds the 10 MB limit. Try a smaller or compressed image.',
  },
  415: {
    title: 'Unsupported file type',
    message: 'The backend does not accept this file type. Use a JPEG, PNG or WebP image.',
  },
  422: {
    title: 'Invalid request',
    message: 'The request was missing something the API needs. Reload the app and try again.',
  },
  429: {
    title: 'Too many requests',
    message: 'The backend is rate limiting requests. Wait a moment, then retry.',
  },
  500: {
    title: 'Detection service error',
    message:
      'The backend failed while processing this image. This is a server-side problem, not your upload — retry in a moment.',
  },
  503: {
    title: 'Detection service unavailable',
    message:
      'The detection models are not ready yet or failed to load. The backend is still starting up — retry in a few seconds.',
  },
}

export function toUserError(error) {
  if (!error) {
    return {
      title: 'Something went wrong',
      message: 'An unexpected problem occurred. Please try again.',
      retryable: true,
      kind: 'unknown',
      status: null,
    }
  }

  // Errors raised deliberately for the user (file validation, cancellation)
  // already carry the exact wording to display.
  if (error.userFacing) {
    return {
      title: error.title,
      message: error.message,
      retryable: Boolean(error.retryable),
      kind: error.kind || 'client',
      status: null,
    }
  }

  const kind = error.kind || 'unknown'
  const status = typeof error.status === 'number' ? error.status : null
  const detail = sanitizeDetail(error.message)

  if (kind === 'config') {
    return {
      title: 'Service not configured',
      message:
        detail ||
        'This frontend build has no API address configured, so detection requests cannot be sent.',
      retryable: false,
      kind,
      status,
    }
  }

  if (kind === 'network') {
    return {
      title: 'Cannot reach the backend',
      message:
        'No response came from the detection API. Check that the backend is running and reachable, then retry.',
      retryable: true,
      kind,
      status,
    }
  }

  if (kind === 'timeout') {
    return {
      title: 'Detection timed out',
      message:
        'The server took too long to answer. The first request after a cold start loads the models and can take a while — retry, or try a smaller image.',
      retryable: true,
      kind,
      status,
    }
  }

  if (kind === 'aborted') {
    return {
      title: 'Analysis cancelled',
      message: 'The detection request was cancelled before it finished.',
      retryable: true,
      kind,
      status,
    }
  }

  if (kind === 'malformed') {
    return {
      title: 'Unexpected server response',
      message:
        'The backend replied with a payload this app cannot read. Nothing was analysed — retry, and check the backend logs if it keeps happening.',
      retryable: true,
      kind,
      status,
    }
  }

  if (kind === 'http' && status !== null) {
    const copy = HTTP_COPY[status] || {
      title: 'Request failed',
      message: 'The backend rejected the detection request.',
    }
    return {
      title: copy.title,
      // Backend `detail` strings here are deliberately user-facing
      // ("Image is larger than the 10 MB demo limit.") and are worth keeping.
      message: detail || copy.message,
      retryable: status >= 500 || status === 404 || status === 429,
      kind,
      status,
    }
  }

  return {
    title: 'Detection failed',
    message: 'Something went wrong while analysing the image. Please try again.',
    retryable: true,
    kind,
    status,
  }
}
