import { useEffect, useRef, useState } from 'react'
import { Video, UploadCloud, Loader2, Play } from 'lucide-react'
import Header from '../components/Header.jsx'
import { Panel } from '../components/Panel.jsx'
import { StatRail } from '../components/StatRail.jsx'
import { api, API_BASE } from '../lib/api'

const PAGE_SIZE = 15

export default function VideoAnalytics() {
  const [summary, setSummary] = useState(null)
  const [error, setError] = useState(null)
  const [detections, setDetections] = useState({ total: 0, results: [] })
  const [page, setPage] = useState(1)

  const [uploadFile, setUploadFile] = useState(null)
  const [job, setJob] = useState(null)
  const [jobError, setJobError] = useState(null)
  const pollRef = useRef(null)
  const fileInputRef = useRef(null)

  useEffect(() => {
    api
      .videoSummary()
      .then(setSummary)
      .catch((e) => setError(e.message))
  }, [])

  useEffect(() => {
    api
      .videoDetections({ page, page_size: PAGE_SIZE })
      .then(setDetections)
      .catch(() => setDetections({ total: 0, results: [] }))
  }, [page])

  useEffect(() => () => clearInterval(pollRef.current), [])

  async function startProcessing() {
    if (!uploadFile) return
    setJobError(null)
    try {
      const { job_id } = await api.processVideo(uploadFile)
      setJob({ job_id, status: 'queued' })
      pollRef.current = setInterval(async () => {
        try {
          const status = await api.processVideoStatus(job_id)
          setJob(status)
          if (status.status === 'completed' || status.status === 'failed') {
            clearInterval(pollRef.current)
          }
        } catch {
          clearInterval(pollRef.current)
        }
      }, 1500)
    } catch (e) {
      setJobError(e.message)
    }
  }

  const stats = summary
    ? [
        { label: 'Resolution', value: `${summary.video.width}×${summary.video.height}` },
        { label: 'Duration', value: `${summary.video.duration_seconds}s` },
        { label: 'Frames sampled', value: summary.processing.processed_frames },
        { label: 'Total detections', value: summary.detection.total_detections },
        { label: 'OCR success', value: summary.detection.ocr_success },
        { label: 'Unique plates', value: summary.detection.unique_plates, tone: 'signal' },
      ]
    : []

  const totalPages = Math.max(1, Math.ceil(detections.total / PAGE_SIZE))

  return (
    <div>
      <Header
        title="Video Analytics"
        description="Stats and the player come from the existing video_results run. Processing a new video is a separate demo job — it does not refresh these charts, the annotated player, or the plate registry, and it does not re-run the 4K source."
      />

      <div className="space-y-6 px-8 py-6">
        {error && (
          <Panel>
            <p className="text-[13px] text-alert">
              No existing video results found ({error}). The saved 4K analytics files are missing.
              A new upload below will not create those dashboard files.
            </p>
          </Panel>
        )}

        {summary && <StatRail stats={stats} />}

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <Panel title="Annotated video" subtitle={summary?.video.filename} bodyClassName="p-3">
            {summary?.annotated_video.available ? (
              <video
                controls
                className="w-full rounded-md border border-hairline-soft bg-black"
                src={`${API_BASE}${summary.annotated_video.url}`}
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
              <div
                onClick={() => fileInputRef.current?.click()}
                className="flex h-24 cursor-pointer items-center justify-center gap-2 rounded-md border border-dashed border-hairline-soft text-center transition-colors hover:border-signal/50"
              >
                <UploadCloud size={18} className="text-ink-faint" strokeWidth={1.5} />
                <span className="text-[12px] text-ink-dim">
                  {uploadFile ? uploadFile.name : 'Click to choose a video file'}
                </span>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/*"
                  className="hidden"
                  onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                />
              </div>

              <button
                onClick={startProcessing}
                disabled={!uploadFile || (job && job.status !== 'completed' && job.status !== 'failed')}
                className="flex w-full items-center justify-center gap-2 rounded-md bg-signal px-4 py-2 text-[13px] font-semibold text-[#062015] transition-colors hover:bg-signal/90 disabled:opacity-50"
              >
                {job && job.status !== 'completed' && job.status !== 'failed' ? (
                  <>
                    <Loader2 size={14} className="animate-spin" />
                    Processing…
                  </>
                ) : (
                  <>
                    <Play size={14} />
                    Process Video
                  </>
                )}
              </button>

              {jobError && <p className="text-[12px] text-alert">{jobError}</p>}

              {job && (
                <div className="rounded-md border border-hairline-soft px-3 py-2 font-mono text-[11px] text-ink-dim">
                  status: {job.status}
                  {job.processed_frames !== undefined && (
                    <> · frames: {job.processed_frames}{job.total_frames ? `/${job.total_frames}` : ''}</>
                  )}
                  {job.detections_so_far !== undefined && <> · detections: {job.detections_so_far}</>}
                  {job.error && <div className="mt-1 text-alert">{job.error}</div>}
                </div>
              )}
            </div>
          </Panel>
        </div>

        <Panel title="Recognized plate observations" subtitle="From detections.csv" bodyClassName="p-0">
          <table className="w-full text-left text-[12px]">
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
              {detections.results.map((d, i) => (
                <tr key={i} className="border-t border-hairline-soft">
                  <td className="px-5 py-2 font-mono text-ink-dim">{d.frame}</td>
                  <td className="px-5 py-2 font-mono text-ink-dim">{d.timestamp}s</td>
                  <td className="px-5 py-2 font-mono text-ink">{d.plate}</td>
                  <td className="px-5 py-2 font-mono text-ink-dim">
                    {(parseFloat(d.yolo_confidence) * 100).toFixed(0)}%
                  </td>
                  <td className="px-5 py-2 font-mono text-ink-dim">
                    {(parseFloat(d.ocr_confidence) * 100).toFixed(0)}%
                  </td>
                  <td className="px-5 py-2 font-mono text-signal">
                    {(parseFloat(d.final_confidence) * 100).toFixed(0)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="flex items-center justify-between border-t border-hairline-soft px-5 py-3 text-[11px] text-ink-dim">
            <span>
              Page {page} of {totalPages} · {detections.total} rows
            </span>
            <div className="flex gap-2">
              <button
                disabled={page <= 1}
                onClick={() => setPage((p) => p - 1)}
                className="rounded border border-hairline px-2 py-1 disabled:opacity-40"
              >
                Prev
              </button>
              <button
                disabled={page >= totalPages}
                onClick={() => setPage((p) => p + 1)}
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
