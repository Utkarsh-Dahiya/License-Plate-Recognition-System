import { useState, useRef, useEffect, useCallback } from 'react'
import {
  api,
  ACCEPTED_IMAGE_TYPES,
  ACCEPTED_IMAGE_EXTENSIONS,
  MAX_IMAGE_BYTES,
  ApiError,
} from '../lib/api'
import { userError } from '../lib/errors'

/**
 * Client-side guard so obviously-wrong uploads fail fast with a readable
 * message instead of uploading and bouncing off the backend. The backend
 * stays the authority — this only mirrors its limits (MAX_IMAGE_BYTES and
 * an `image/*` content-type check in app/backend/main.py).
 */
function validateImageFile(file) {
  if (!file) {
    return userError({
      title: 'No file selected',
      message: 'Choose an image before running detection.',
    })
  }

  const type = (file.type || '').toLowerCase()
  const name = (file.name || '').toLowerCase()
  const looksKnown =
    ACCEPTED_IMAGE_TYPES.includes(type) ||
    ACCEPTED_IMAGE_EXTENSIONS.some((ext) => name.endsWith(ext))

  // Some browsers report an empty type for dragged files; only reject when
  // there is positive evidence the file is not a supported image.
  if (!looksKnown) {
    return userError({
      title: 'Unsupported file type',
      message: 'Choose a JPEG, PNG or WebP image. Other file formats cannot be analysed.',
    })
  }

  if (file.size === 0) {
    return userError({
      title: 'File is empty',
      message: 'That file contains no data. Try exporting or copying the image again.',
    })
  }

  if (file.size > MAX_IMAGE_BYTES) {
    return userError({
      title: 'Image is too large',
      message: 'That image is larger than the 10 MB limit. Compress it or pick a smaller photo.',
    })
  }

  return null
}

/**
 * Read intrinsic dimensions from an object URL. Purely local (no request):
 * used for the preview caption and to fail early on a file the browser
 * cannot decode at all.
 */
function readImageDimensions(url) {
  return new Promise((resolve) => {
    const img = new Image()
    img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight })
    img.onerror = () => resolve(null)
    img.src = url
  })
}

/**
 * Guard against a malformed payload. The backend contract is stable
 * (image / plates_detected / plates / annotated_image_base64 /
 * processing_time_seconds) but the UI must not crash if a proxy, a stale
 * deploy or a future breaking change returns something else.
 */
function validateDetectionPayload(payload) {
  if (!payload || typeof payload !== 'object') {
    return userError({
      title: 'Unexpected server response',
      message: 'The backend replied with an empty or unreadable payload, so there is nothing to display.',
    })
  }
  if (!Array.isArray(payload.plates)) {
    return userError({
      title: 'Unexpected server response',
      message:
        'The detection response did not include a list of plates. The API and this frontend may be out of sync.',
    })
  }
  if (typeof payload.plates_detected !== 'number') {
    return userError({
      title: 'Unexpected server response',
      message: 'The detection response did not include a plate count.',
    })
  }
  return null
}

export function useDetection() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [fileMeta, setFileMeta] = useState(null)
  const [confThreshold, setConfThreshold] = useState(0.20)
  const [status, setStatus] = useState('idle') // 'idle' | 'ready' | 'processing' | 'success' | 'error'
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [hoveredPlateId, setHoveredPlateId] = useState(null)
  const [selectedPlateId, setSelectedPlateId] = useState(null)
  const [selectedSampleId, setSelectedSampleId] = useState(null)
  const [sampleStatus, setSampleStatus] = useState('idle') // 'idle' | 'loading'
  const [warmingUp, setWarmingUp] = useState(false)
  const [elapsedSeconds, setElapsedSeconds] = useState(0)

  const inFlightRef = useRef(false)
  const detectAbortRef = useRef(null)
  const sampleAbortRef = useRef(null)
  const previewUrlRef = useRef(null)
  const elapsedRef = useRef(null)

  /** Replace the preview URL, revoking the previous one to avoid leaks. */
  const applyPreview = useCallback((nextUrl) => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current)
    }
    previewUrlRef.current = nextUrl
    setPreviewUrl(nextUrl)
  }, [])

  const stopTimer = useCallback(() => {
    if (elapsedRef.current) {
      clearInterval(elapsedRef.current)
      elapsedRef.current = null
    }
    setElapsedSeconds(0)
  }, [])

  /** Cancel anything in flight and stop derived state. */
  const cancelInFlight = useCallback(() => {
    detectAbortRef.current?.abort()
    detectAbortRef.current = null
    sampleAbortRef.current?.abort()
    sampleAbortRef.current = null
    inFlightRef.current = false
    stopTimer()
  }, [stopTimer])

  // Poll backend readiness on mount to notify if models are still warming up
  useEffect(() => {
    let cancelled = false
    let attempts = 0
    let timer = null

    async function checkReadiness() {
      try {
        const r = await api.readiness()
        if (cancelled) return

        if (r.models_loaded || r.status === 'ready') {
          setWarmingUp(false)
          clearInterval(timer)
          return
        }
        if (r.status === 'error') {
          setWarmingUp(false)
          clearInterval(timer)
          return
        }
        setWarmingUp(true)
      } catch {
        if (!cancelled) setWarmingUp(false)
      }
    }

    checkReadiness()
    timer = setInterval(() => {
      attempts += 1
      if (attempts > 30) {
        clearInterval(timer)
        return
      }
      checkReadiness()
    }, 2500)

    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [])

  // Release the object URL and any pending request when leaving the page.
  useEffect(
    () => () => {
      detectAbortRef.current?.abort()
      sampleAbortRef.current?.abort()
      if (elapsedRef.current) clearInterval(elapsedRef.current)
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current)
    },
    [],
  )

  const selectFile = useCallback(
    (newFile) => {
      if (!newFile) return

      const invalid = validateImageFile(newFile)
      if (invalid) {
        // A rejected upload must not disturb an image already on screen.
        setError(invalid)
        setStatus('error')
        return
      }

      // A manual upload supersedes any sample and cancels pending work.
      cancelInFlight()
      setFile(newFile)
      setResult(null)
      setError(null)
      setSelectedPlateId(null)
      setHoveredPlateId(null)
      setSelectedSampleId(null)
      setSampleStatus('idle')
      setStatus('ready')

      const url = URL.createObjectURL(newFile)
      applyPreview(url)
      setFileMeta({
        name: newFile.name || 'uploaded_image.jpg',
        size: newFile.size ? `${(newFile.size / 1024).toFixed(1)} KB` : '—',
        type: newFile.type || 'image/jpeg',
        source: 'upload',
      })

      // Dimensions are nice-to-have metadata: resolve them asynchronously
      // without gating the "ready" state.
      readImageDimensions(url).then((dims) => {
        if (previewUrlRef.current !== url) return
        if (!dims) {
          setError(
            userError({
              title: 'Image could not be read',
              message:
                'This file could not be decoded as an image. It may be corrupt — try re-exporting it.',
            }),
          )
          setStatus('error')
          return
        }
        setFileMeta((prev) => (prev ? { ...prev, ...dims } : prev))
      })
    },
    [applyPreview, cancelInFlight],
  )

  /**
   * Curated-sample selection is pure React state: `selectedSampleId` is set
   * synchronously on click and the resulting File becomes the detection
   * target. No DOM queries, no reliance on element order.
   */
  const selectSample = useCallback(
    async (sample) => {
      if (!sample || sample.available === false) return

      // Last click wins: cancel anything in flight BEFORE arming the new
      // controller. cancelInFlight() aborts sampleAbortRef.current, so
      // creating the new controller first would abort the request we are
      // about to issue (it would reject instantly and never load).
      cancelInFlight()
      sampleAbortRef.current?.abort()
      const controller = new AbortController()
      sampleAbortRef.current = controller

      setSelectedSampleId(sample.id)
      setSampleStatus('loading')
      setResult(null)
      setError(null)
      setSelectedPlateId(null)
      setHoveredPlateId(null)
      setStatus('idle')

      try {
        const blob = await api.fetchSampleBlob(sample.id, { signal: controller.signal })
        if (controller.signal.aborted) return

        const sampleFile = new File([blob], sample.filename || `${sample.id}.jpg`, {
          type: blob.type || 'image/jpeg',
        })

        setFile(sampleFile)
        const url = URL.createObjectURL(blob)
        applyPreview(url)
        setFileMeta({
          name: sample.filename || `${sample.id}.jpg`,
          size: sample.size_bytes
            ? `${(sample.size_bytes / 1024).toFixed(1)} KB`
            : `${(blob.size / 1024).toFixed(1)} KB`,
          type: blob.type || 'image/jpeg',
          sampleTitle: sample.title,
          source: 'sample',
        })
        setSampleStatus('idle')
        setStatus('ready')

        readImageDimensions(url).then((dims) => {
          if (previewUrlRef.current !== url || !dims) return
          setFileMeta((prev) => (prev ? { ...prev, ...dims } : prev))
        })
      } catch (err) {
        if (controller.signal.aborted || (err instanceof ApiError && err.kind === 'aborted')) {
          // Superseded by a newer selection (or unmounted). Never leave the
          // gallery stuck on a spinner when this controller is still current.
          if (sampleAbortRef.current === controller) {
            sampleAbortRef.current = null
            setSampleStatus('idle')
          }
          return
        }
        // Selection is authoritative state: on failure, drop back to "no
        // sample selected" so the UI never claims a selection with no image.
        setSampleStatus('idle')
        setSelectedSampleId(null)
        setError({
          userFacing: true,
          title: 'Sample could not be loaded',
          message:
            err?.kind === 'network' || err?.kind === 'timeout'
              ? 'The sample image could not be downloaded from the backend. Check the connection and try again.'
              : err?.message || 'The sample image could not be loaded. Try another sample.',
          retryable: true,
          kind: err?.kind || 'client',
        })
        setStatus('error')
      }
    },
    [applyPreview, cancelInFlight],
  )

  /**
   * @param {number} [confOverride] run at this threshold instead of the
   *   current slider value. Needed because "re-run with a lower threshold"
   *   changes the slider and starts a run in the same tick, where the
   *   callback would still close over the previous value.
   */
  const runDetection = useCallback(async (confOverride) => {
    // Guard against duplicate submissions (double click, Enter key repeat,
    // sample click while a run is in flight).
    if (!file || inFlightRef.current) return

    const conf = typeof confOverride === 'number' ? confOverride : confThreshold
    const controller = new AbortController()
    detectAbortRef.current = controller
    inFlightRef.current = true

    setStatus('processing')
    setError(null)
    setSelectedPlateId(null)
    setHoveredPlateId(null)

    // Elapsed time is real measured time — never a fabricated percentage or
    // a fake stage sequence the backend does not report.
    const startedAt = Date.now()
    setElapsedSeconds(0)
    elapsedRef.current = setInterval(() => {
      setElapsedSeconds((Date.now() - startedAt) / 1000)
    }, 250)

    try {
      const res = await api.detectImage(file, conf, { signal: controller.signal })

      const malformed = validateDetectionPayload(res)
      if (malformed) {
        setError(malformed)
        setStatus('error')
        return
      }

      setResult(res)
      setStatus('success')

      // Pre-select the strongest plate so the canvas and the result list are
      // immediately in sync. Deterministic: ties fall back to plate_id order.
      if (res.plates.length > 0) {
        const best = [...res.plates].sort((a, b) => {
          const diff =
            (Number(b.final_confidence) || 0) - (Number(a.final_confidence) || 0)
          return diff !== 0 ? diff : (a.plate_id ?? 0) - (b.plate_id ?? 0)
        })[0]
        setSelectedPlateId(best.plate_id)
      }
    } catch (err) {
      // A cancel is a user action, not a failure worth an error banner.
      if (controller.signal.aborted || err?.kind === 'aborted') {
        setStatus('ready')
        return
      }
      // Keep the raw error; the page maps it through toUserError() so the
      // same taxonomy also covers locally-raised validation errors.
      setError(err)
      setStatus('error')
    } finally {
      if (detectAbortRef.current === controller) detectAbortRef.current = null
      inFlightRef.current = false
      stopTimer()
    }
  }, [file, confThreshold, stopTimer])

  const reset = useCallback(() => {
    cancelInFlight()
    setFile(null)
    setFileMeta(null)
    setResult(null)
    setError(null)
    setStatus('idle')
    setSelectedPlateId(null)
    setHoveredPlateId(null)
    setSelectedSampleId(null)
    setSampleStatus('idle')
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current)
      previewUrlRef.current = null
    }
    setPreviewUrl(null)
  }, [cancelInFlight])

  return {
    file,
    previewUrl,
    fileMeta,
    confThreshold,
    setConfThreshold,
    status,
    result,
    error,
    hoveredPlateId,
    setHoveredPlateId,
    selectedPlateId,
    setSelectedPlateId,
    selectedSampleId,
    sampleStatus,
    warmingUp,
    elapsedSeconds,
    isProcessing: status === 'processing',
    selectFile,
    selectSample,
    runDetection,
    reset,
  }
}
