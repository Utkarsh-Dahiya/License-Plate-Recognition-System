import { useEffect, useRef, useState } from 'react'
import { Video, UploadCloud, Loader2, Play } from 'lucide-react'
import Header from '../components/Header.jsx'
import { Panel } from '../components/Panel.jsx'
import { StatRail } from '../components/StatRail.jsx'
import ErrorNotice from '../components/ErrorNotice.jsx'
import { api, API_BASE, MAX_VIDEO_BYTES } from '../lib/api'
import { userError } from '../lib/errors'
import { useAsyncResource } from '../hooks/useAsyncResource'

const PAGE_SIZE = 15
const POLL_INTERVAL_MS = 1500
// A slow backend must not look like a dead job: only a run of failed polls
// counts as losing it, and each blip is rescheduled instead of dropped.
const MAX_CONSECUTIVE_POLL_FAILURES = 5
// Mirrors the `video/*` rule and the 50 MB cap in app/backend/main.py so a bad
// file is rejected before a single byte is uploaded.
const VIDEO_EXTENSIONS = [
  '.mp4',
  '.webm',
  '.mov',
  '.avi',
  '.mkv',
  '.m4v',
  '.mpeg',
  '.mpg',
  '.3gp',
]

/** '91' for a 0-1 ratio, '—' when the field is missing or malformed. */
function formatRatioPercent(value) {
  const n = Number(value)
  return Number.isFinite(n) ? `${(n * 100).toFixed(0)}%` : '—'
}

function formatMegabytes(bytes) {
  const n = Number(bytes)
  if (!Number.isFinite(n) || n <= 0) return '0 MB'
  return `${(n / (1024 * 1024)).toFixed(1)} MB`
}

/** Client-side mirror of the backend's upload rules. Returns null when ok. */
function validateVideoFile(file) {
  if (!file) {
    return userError({
      title: 'No video chosen',
      message: 'Pick a short clip before starting processing.',
    })
  }

  const type = (file.type || '').toLowerCase()
  const name = (file.name || '').toLowerCase()
  const hasVideoExtension = VIDEO_EXTENSIONS.some((ext) => name.endsWith(ext))

  if (!type.startsWith('video/') && !hasVideoExtension) {
    return userError({
      title: 'Unsupported file',
      message: 'That is not a video file. Choose a clip such as mp4, mov or webm.',
    })
  }

  if (file.size === 0) {
    return userError({
      title: 'Empty file',
      message: 'The selected file is 0 bytes, so there is nothing to process.',
    })
  }

  if (file.size > MAX_VIDEO_BYTES) {
    return userError({
      title: 'Video too large',
      message: 'The server demo limit is 50 MB. Trim or re-encode the clip and try again.',
    })
  }

  return null
}

export default function VideoAnalytics() {
  const [page, setPage] = useState(1)

  // Both reads carry loading / error / retry and drop out-of-order replies.
  const summary = useAsyncResource(() => api.videoSummary(), [])

  const detections = useAsyncResource(
    () => api.videoDetections({ page, page_size: PAGE_SIZE }),
    [page],
    { initialData: { total: 0, results: [] } },
  )

  const [uploadFile, setUploadFile] = useState(null)
  const [fileError, setFileError] = useState(null)
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState(null)
  const [job, setJob] = useState(null)
  const [jobError, setJobError] = useState(null)

  const pollRef = useRef(null)
  const fileInputRef = useRef(null)
  const submittingRef = useRef(false)
  const pollFailuresRef = useRef(0)
  const mountedRef = useRef(true)

  // Sole owner of the status timer: cleared before every reschedule and on
  // unmount, so polls never stack and no setState lands after leaving.
  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      clearTimeout(pollRef.current)
    }
  }, [])

  function scheduleNextPoll(jobId, delayMs) {
    clearTimeout(pollRef.current)
    if (!mountedRef.current) return
    pollRef.current = setTimeout(() => pollJob(jobId), delayMs)
  }

  async function pollJob(jobId) {
    if (!mountedRef.current) return
    try {
      const status = await api.processVideoStatus(jobId)
      if (!mountedRef.current) return
      pollFailuresRef.current = 0
      setJob(status)
      if (status.status !== 'completed' && status.status !== 'failed') {
        // Self-scheduling instead of setInterval: the next poll starts only
        // after this response, so a stalled job cannot queue up requests.
        scheduleNextPoll(jobId, POLL_INTERVAL_MS)
      }
    } catch {
      if (!mountedRef.current) return
      pollFailuresRef.current += 1
      if (pollFailuresRef.current >= MAX_CONSECUTIVE_POLL_FAILURES) {
        // Keep the last good readout and explain, instead of freezing the
        // job line silently (the old behaviour).
        setJobError(
          userError({
            title: 'Lost contact with the job',
            message:
              'The status endpoint stopped answering, so progress is no longer updating. The job may still be running — reconnect to resume polling.',
            retryable: true,
          }),
        )
        return
      }
      scheduleNextPoll(jobId, POLL_INTERVAL_MS)
    }
  }

  function resumePolling() {
    if (!job || job.status === 'completed' || job.status === 'failed') return
    setJobError(null)
    pollFailuresRef.current = 0
    scheduleNextPoll(job.job_id, 0)
  }

  function handleFilePicked(file) {
    const problem = validateVideoFile(file)
    if (problem) {
      setUploadFile(null)
      setFileError(problem)
      return
    }
    setUploadFile(file)
    setFileError(null)
    setSubmitError(null)
  }

  async function startProcessing() {
    // Repeat clicks and double submits are ignored while one is in flight.
    if (submittingRef.current) return

    const problem = validateVideoFile(uploadFile)
    if (problem) {
      setFileError(problem)
      return
    }

    submittingRef.current = true
    setSubmitting(true)
    setFileError(null)
    setSubmitError(null)
    setJobError(null)

    try {
      const { job_id } = await api.processVideo(uploadFile)
      if (!mountedRef.current) return
      clearTimeout(pollRef.current)
      pollFailuresRef.current = 0
      setJob({ job_id, status: 'queued' })
      scheduleNextPoll(job_id, POLL_INTERVAL_MS)
    } catch (err) {
      if (!mountedRef.current) return
      setSubmitError(err)
    } finally {
      submittingRef.current = false
      if (mountedRef.current) setSubmitting(false)
    }
  }

  const dash = summary.data
  const stats = dash
    ? [
        {
          label: 'Resolution',
          value: `${dash.video?.width ?? '—'}×${dash.video?.height ?? '—'}`,
        },
        {
          label: 'Duration',
          value:
            dash.video?.duration_seconds !== undefined
              ? `${dash.video.duration_seconds}s`
              : '—',
        },
        { label: 'Frames sampled', value: dash.processing?.processed_frames ?? '—' },
        { label: 'Total detections', value: dash.detection?.total_detections ?? '—' },
        { label: 'OCR success', value: dash.detection?.ocr_success ?? '—' },
        {
          label: 'Unique plates',
          value: dash.detection?.unique_plates ?? '—',
          tone: 'signal',
        },
      ]
    : []

  const results = detections.data.results ?? []
  const totalRows = detections.data.total ?? 0
  const totalPages = Math.max(1, Math.ceil(totalRows / PAGE_SIZE))
  const jobActive = Boolean(job) && job.status !== 'completed' && job.status !== 'failed'

  return (
    <div>
      <Header
        title="Video Analytics"
        description="Stats and the player come from the existing video_results run. Processing a new video is a separate demo job — it does not refresh these charts, the annotated player, or the plate registry, and it does not re-run the 4K source."
      />

      <div className="space-y-6 px-8 py-6">
        {/* A missing/stale summary is announced with a retry, not a dead end. */}
        {summary.error && (
          <div className="space-y-2">
            <ErrorNotice
              error={summary.error}
              onRetry={summary.retry}
              retryLabel="Reload summary"
            />

            <p className="text-[12px] leading-5 text-ink-dim">
              The saved 4K analytics files in{' '}
              <span className="font-mono text-[11px]">video_results/</span> feed the stats,
              player and registry below — a new upload does not create them.
            </p>
          </div>
        )}

        {summary.data ? (
          <StatRail stats={stats} />
        ) : !summary.error ? (
          <div
            role="status"
            className="h-24 animate-pulse rounded-lg border border-hairline bg-panel"
          >
            <span className="sr-only">Loading video summary…</span>
          </div>
        ) : null}

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <Panel
            title="Annotated video"
            subtitle={summary.data?.video?.filename}
            bodyClassName="p-3"
          >
            {summary.loading && !summary.data ? (
              <div
                role="status"
                className="flex h-40 animate-pulse items-center justify-center rounded-md border border-hairline-soft bg-panel-raised"
              >
                <span className="sr-only">Loading annotated video…</span>
              </div>
            ) : summary.error && !summary.data ? (
              <div className="flex h-40 items-center justify-center rounded-md border border-hairline-soft bg-panel-raised px-6 text-center">
                <p className="max-w-xs text-[12px] text-ink-faint">
                  The annotated player could not be loaded. Reload the summary above to try
                  again.
                </p>
              </div>
            ) : summary.data?.annotated_video?.available &&
              summary.data?.annotated_video?.url ? (
              <video
                controls
                aria-label="Annotated video preview"
                className="w-full rounded-md border border-hairline-soft bg-black"
                src={`${API_BASE}${summary.data.annotated_video.url}`}
              />
            ) : (
              <div className="flex h-40 flex-col items-center justify-center gap-2 rounded-md border border-hairline-soft bg-panel-raised text-center text-ink-faint">
                <Video size={20} strokeWidth={1.5} />
                <p className="max-w-xs text-[12px]">
                  annotated_video.mp4 wasn't found at{' '}
                  <span className="font-mono text-[11px]">video_results/</span>. Drop the real
                  file there (or set <span className="font-mono text-[11px]">LVA_ANNOTATED_VIDEO</span>)
                  and it'll play here — no rebuild needed.
                </p>
              </div>
            )}
          </Panel>

          <Panel title="Process a new video" subtitle="Demo job only — does not update saved 4K stats, player, or registry">
            <div className="space-y-3">
              <p className="text-[12px] leading-5 text-ink-dim">
                Use a short clip. This path will not replace video_results CSV/JSON or the annotated
                4K file. Live processing is capped on the server for demo safety.
              </p>
              {/* A real button keeps the picker keyboard-reachable; the input
                  stays hidden and out of the tab order. */}
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                aria-label={
                  uploadFile
                    ? `Selected file ${uploadFile.name}. Choose a different video`
                    : 'Choose a video file to process'
                }
                className="flex h-24 w-full cursor-pointer items-center justify-center gap-2 rounded-md border border-dashed border-hairline-soft text-center transition-colors hover:border-signal/50 focus:outline-none focus-visible:border-signal/60 focus-visible:ring-1 focus-visible:ring-signal/40"
              >
                <UploadCloud
                  size={18}
                  className="text-ink-faint"
                  strokeWidth={1.5}
                  aria-hidden="true"
                />
                <span className="text-[12px] text-ink-dim">
                  {uploadFile
                    ? `${uploadFile.name} · ${formatMegabytes(uploadFile.size)}`
                    : 'Click to choose a video file'}
                </span>
              </button>

              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                aria-hidden="true"
                tabIndex={-1}
                className="hidden"
                onChange={(e) => {
                  const picked = e.target.files?.[0] || null
                  // Reset first, so the same file can be re-picked after it
                  // was rejected by validation.
                  e.target.value = ''
                  if (picked) handleFilePicked(picked)
                }}
              />

              {fileError && (
                <ErrorNotice error={fileError} onDismiss={() => setFileError(null)} />
              )}

              <button
                type="button"
                onClick={startProcessing}
                disabled={!uploadFile || submitting || jobActive}
                className="flex w-full items-center justify-center gap-2 rounded-md bg-signal px-4 py-2 text-[13px] font-semibold text-[#062015] transition-colors hover:bg-signal/90 disabled:opacity-50"
              >
                {submitting || jobActive ? (
                  <>
                    <Loader2 size={14} className="animate-spin" aria-hidden="true" />
                    {submitting ? 'Uploading…' : 'Processing…'}
                  </>
                ) : (
                  <>
                    <Play size={14} aria-hidden="true" />
                    Process Video
                  </>
                )}
              </button>

              {submitError && (
                <ErrorNotice
                  error={submitError}
                  onRetry={startProcessing}
                  retryLabel="Retry submission"
                  onDismiss={() => setSubmitError(null)}
                />
              )}

              {job && (
                <div
                  role="status"
                  className="rounded-md border border-hairline-soft px-3 py-2 font-mono text-[11px] text-ink-dim"
                >
                  status: {job.status}
                  {job.processed_frames !== undefined && (
                    <>
                      {' '}
                      · frames: {job.processed_frames}
                      {job.total_frames ? `/${job.total_frames}` : ''}
                    </>
                  )}
                  {job.detections_so_far !== undefined && (
                    <> · detections: {job.detections_so_far}</>
                  )}
                  {job.error && <div className="mt-1 text-alert">{job.error}</div>}
                </div>
              )}

              {jobError && (
                <ErrorNotice
                  error={jobError}
                  onRetry={resumePolling}
                  retryLabel="Reconnect"
                  onDismiss={() => setJobError(null)}
                />
              )}
            </div>
          </Panel>
        </div>

        <Panel title="Recognized plate observations" subtitle="From detections.csv" bodyClassName="p-0">
          {/* A failed page load is reported instead of looking empty, and the
              pager below stays usable to move away from it. */}
          {detections.error && (
            <div className="px-5 py-4">
              <ErrorNotice
                error={detections.error}
                onRetry={detections.retry}
                retryLabel="Reload observations"
              />
            </div>
          )}

          {/* Scrollable region keeps all six columns readable at 390px. */}
          <div
            className="overflow-x-auto"
            role="region"
            aria-label="Recognized plate observations"
            aria-busy={detections.loading}
            tabIndex={0}
          >
            <table className="w-full min-w-[640px] text-left text-[12px]">
              <thead>
                <tr className="text-ink-faint">
                  <th className="px-5 py-2 font-medium">Frame</th>
                  <th className="px-5 py-2 font-medium">Timestamp</th>
                  <th className="px-5 py-2 font-medium">Plate</th>
                  <th className="px-5 py-2 font-medium">YOLO</th>
                  <th className="px-5 py-2 font-medium">OCR</th>
                  <th className="px-5 py-2 font-medium">Final</th>
                </tr>
              </thead>
              <tbody>
                {detections.loading && results.length === 0 && (
                  <tr>
                    <td colSpan={6} className="px-5 py-8 text-center">
                      <span
                        role="status"
                        className="inline-flex items-center gap-2 text-ink-faint"
                      >
                        <Loader2 size={13} className="animate-spin" aria-hidden="true" />
                        Loading observations…
                      </span>
                    </td>
                  </tr>
                )}

                {results.map((d, i) => (
                  <tr
                    key={`${d.plate ?? 'row'}-${d.frame ?? i}-${i}`}
                    className="border-t border-hairline-soft"
                  >
                    <td className="px-5 py-2 font-mono text-ink-dim">{d.frame}</td>
                    <td className="px-5 py-2 font-mono text-ink-dim">{d.timestamp}s</td>
                    <td className="px-5 py-2 font-mono text-ink">{d.plate}</td>
                    <td className="px-5 py-2 font-mono text-ink-dim">
                      {formatRatioPercent(d.yolo_confidence)}
                    </td>
                    <td className="px-5 py-2 font-mono text-ink-dim">
                      {formatRatioPercent(d.ocr_confidence)}
                    </td>
                    <td className="px-5 py-2 font-mono text-signal">
                      {formatRatioPercent(d.final_confidence)}
                    </td>
                  </tr>
                ))}

                {!detections.loading && !detections.error && results.length === 0 && (
                  <tr>
                    <td colSpan={6} className="px-5 py-8 text-center text-ink-faint">
                      <p role="status">
                        {totalRows === 0
                          ? 'No plate observations have been recorded yet.'
                          : 'This page has no rows — go back to an earlier page.'}
                      </p>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <div className="flex items-center justify-between border-t border-hairline-soft px-5 py-3 text-[11px] text-ink-dim">
            <span role="status">
              Page {page} of {totalPages} · {totalRows} rows
              {detections.loading && results.length > 0 ? ' · updating…' : ''}
            </span>
            <div className="flex gap-2">
              <button
                type="button"
                disabled={page <= 1 || detections.loading}
                onClick={() => setPage((p) => p - 1)}
                aria-label="Previous page"
                className="rounded border border-hairline px-2 py-1 disabled:opacity-40"
              >
                Prev
              </button>
              <button
                type="button"
                disabled={page >= totalPages || detections.loading}
                onClick={() => setPage((p) => p + 1)}
                aria-label="Next page"
                className="rounded border border-hairline px-2 py-1 disabled:opacity-40"
              >
                Next
              </button>
            </div>
          </div>
        </Panel>
      </div>
    </div>
  )
}
