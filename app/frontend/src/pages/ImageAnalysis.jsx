import { useRef, useState } from 'react'
import { UploadCloud, Loader2, RotateCcw, ShieldCheck, AlertTriangle, CircleAlert } from 'lucide-react'
import Header from '../components/Header.jsx'
import { Panel } from '../components/Panel.jsx'
import StatusBadge from '../components/StatusBadge.jsx'
import { api } from '../lib/api'

const STEPS = [
  'Loading YOLO...',
  'Detecting license plates...',
  'Running OCR...',
  'Scoring result...',
]

function getConfidenceLevel(plate) {
  const confidence = Number(plate?.final_confidence ?? 0)

  if (confidence >= 0.8) {
    return {
      label: 'HIGH CONFIDENCE',
      tone: 'high',
      icon: ShieldCheck,
    }
  }

  if (confidence >= 0.5) {
    return {
      label: 'NEEDS REVIEW',
      tone: 'review',
      icon: AlertTriangle,
    }
  }

  return {
    label: 'LOW CONFIDENCE',
    tone: 'low',
    icon: CircleAlert,
  }
}

function ConfidenceIndicator({ plate }) {
  const level = getConfidenceLevel(plate)
  const Icon = level.icon
  const percentage = Math.max(
    0,
    Math.min(100, Number(plate?.final_confidence ?? 0) * 100),
  )

  const toneClasses = {
    high: 'border-signal-dim bg-signal-dim/10 text-signal',
    review: 'border-amber-400/20 bg-amber-400/5 text-amber-300',
    low: 'border-alert/20 bg-alert/5 text-alert',
  }

  const barClasses = {
    high: 'bg-signal',
    review: 'bg-amber-400',
    low: 'bg-alert',
  }

  return (
    <div
      className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-[10px] font-semibold tracking-wider ${toneClasses[level.tone]}`}
    >
      <Icon size={12} />
      {level.label}
    </div>
  )
}

export default function ImageAnalysis() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [loading, setLoading] = useState(false)
  const [stepIndex, setStepIndex] = useState(0)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)

  function handleFile(f) {
    if (!f) return
    setFile(f)
    setPreviewUrl(URL.createObjectURL(f))
    setResult(null)
    setError(null)
  }

  async function runDetection() {
    if (!file) return

    setLoading(true)
    setError(null)
    setStepIndex(0)

    const stepTimer = setInterval(() => {
      setStepIndex((i) => Math.min(i + 1, STEPS.length - 1))
    }, 900)

    try {
      const res = await api.detectImage(file)
      setResult(res)
    } catch (e) {
      if (e.status === 503) {
        setError(
          'The detection model is unavailable right now — best.pt could not be loaded on the server. Check LVA_MODEL_PATH and the System page.',
        )
      } else if (e.status === 400) {
        setError(e.message || 'That file could not be processed as an image.')
      } else {
        setError(e.message || 'Something went wrong reaching the backend.')
      }
    } finally {
      clearInterval(stepTimer)
      setLoading(false)
    }
  }

  function reset() {
    setFile(null)
    setPreviewUrl(null)
    setResult(null)
    setError(null)
  }

  const bestPlate = result?.plates?.length
    ? [...result.plates].sort(
        (a, b) => b.final_confidence - a.final_confidence,
      )[0]
    : null

  return (
    <div>
      <Header
        title="Live / Image Analysis"
        description="Upload an image and run it through the real YOLO + EasyOCR pipeline."
      />

      <div className="space-y-6 px-8 py-6">
        {!previewUrl && (
          <div
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault()
              handleFile(e.dataTransfer.files?.[0])
            }}
            className="flex cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed border-hairline bg-panel px-8 py-20 text-center transition-colors hover:border-signal/50"
          >
            <UploadCloud
              size={22}
              className="mb-4 text-ink-faint"
              strokeWidth={1.5}
            />

            <p className="text-[13px] text-ink">
              Drop an image here, or click to browse
            </p>

            <p className="mt-1 text-[12px] text-ink-faint">
              JPG, PNG, WEBP
            </p>

            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => handleFile(e.target.files?.[0])}
            />
          </div>
        )}

        {previewUrl && (
          <>
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              <Panel title="Original image" bodyClassName="p-3">
                <img
                  src={previewUrl}
                  alt="Original upload"
                  className="w-full rounded-md border border-hairline-soft object-contain"
                />
              </Panel>

              <Panel
                title="Annotated result"
                subtitle={
                  result
                    ? 'YOLO bounding boxes + best OCR read'
                    : undefined
                }
                bodyClassName="p-3"
              >
                {result?.annotated_image_base64 ? (
                  <img
                    src={`data:image/jpeg;base64,${result.annotated_image_base64}`}
                    alt="Annotated result"
                    className="w-full rounded-md border border-hairline-soft object-contain"
                  />
                ) : (
                  <div className="flex h-full min-h-[160px] items-center justify-center rounded-md border border-dashed border-hairline-soft text-center text-[12px] text-ink-faint">
                    {loading
                      ? STEPS[stepIndex]
                      : 'Run detection to see bounding boxes here.'}
                  </div>
                )}
              </Panel>
            </div>

            <div className="flex items-center gap-2">
              {!result && (
                <button
                  onClick={runDetection}
                  disabled={loading}
                  className="flex items-center gap-2 rounded-md bg-signal px-4 py-2 text-[13px] font-semibold text-[#062015] transition-colors hover:bg-signal/90 disabled:opacity-60"
                >
                  {loading && (
                    <Loader2 size={14} className="animate-spin" />
                  )}

                  {loading ? STEPS[stepIndex] : 'Run Detection'}
                </button>
              )}

              <button
                onClick={reset}
                className="flex items-center gap-2 rounded-md border border-hairline px-4 py-2 text-[13px] text-ink-dim transition-colors hover:bg-panel-raised"
              >
                <RotateCcw size={14} />
                Analyze another image
              </button>
            </div>

            {error && (
              <Panel>
                <p className="text-[13px] leading-relaxed text-alert">
                  {error}
                </p>
              </Panel>
            )}

            {result && (
              <Panel title="Detection results">
                <div className="mb-5 flex flex-col gap-2 border-b border-hairline-soft pb-4 sm:flex-row sm:items-center sm:justify-between">
                  <div>
                    <div className="text-[13px] font-medium text-ink">
                      {result.plates_detected} plate
                      {result.plates_detected === 1 ? '' : 's'} detected
                    </div>

                    <div className="mt-1 text-[11px] text-ink-faint">
                      Results are classified by observed confidence and may
                      require human review.
                    </div>
                  </div>

                  <span className="font-mono text-[11px] text-ink-dim">
                    {result.processing_time_seconds}s · YOLO · EasyOCR
                  </span>
                </div>

                {result.plates.length === 0 && (
                  <div className="rounded-md border border-dashed border-hairline-soft p-6 text-center">
                    <p className="text-[13px] text-ink-dim">
                      No plate detected in this image.
                    </p>
                  </div>
                )}

                <div className="space-y-4">
                  {result.plates.map((p) => {
                    const confidenceLevel = getConfidenceLevel(p)
                    const percentage = Math.max(
                      0,
                      Math.min(100, Number(p.final_confidence ?? 0) * 100),
                    )

                    const isBest = bestPlate?.plate_id === p.plate_id

                    return (
                      <div
                        key={p.plate_id}
                        className={`overflow-hidden rounded-lg border ${
                          isBest
                            ? 'border-signal-dim bg-signal-dim/5'
                            : 'border-hairline-soft bg-panel'
                        }`}
                      >
                        <div className="flex flex-col gap-4 p-4 sm:flex-row sm:items-start sm:justify-between">
                          <div>
                            <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-ink-faint">
                              Detected plate #{p.plate_id}
                            </div>

                            <div className="font-mono text-2xl font-semibold tracking-[0.12em] text-ink">
                              {p.ocr_text || 'UNKNOWN'}
                            </div>

                            <div className="mt-2 text-[11px] text-ink-faint">
                              OCR output from EasyOCR
                            </div>
                          </div>

                          <div className="flex flex-col items-start gap-2 sm:items-end">
                            <ConfidenceIndicator plate={p} />
                            <StatusBadge status={p.status} />
                          </div>
                        </div>

                        <div className="grid grid-cols-1 border-y border-hairline-soft sm:grid-cols-3">
                          <div className="border-b border-hairline-soft p-4 text-center sm:border-b-0 sm:border-r">
                            <div className="font-mono text-lg text-ink">
                              {(p.yolo_confidence * 100).toFixed(1)}%
                            </div>

                            <div className="mt-1 text-[10px] uppercase tracking-wider text-ink-faint">
                              YOLO Confidence
                            </div>
                          </div>

                          <div className="border-b border-hairline-soft p-4 text-center sm:border-b-0 sm:border-r">
                            <div className="font-mono text-lg text-ink">
                              {(p.ocr_confidence * 100).toFixed(1)}%
                            </div>

                            <div className="mt-1 text-[10px] uppercase tracking-wider text-ink-faint">
                              OCR Confidence
                            </div>
                          </div>

                          <div className="p-4 text-center">
                            <div className="font-mono text-lg text-signal">
                              {(p.final_confidence * 100).toFixed(1)}%
                            </div>

                            <div className="mt-1 text-[10px] uppercase tracking-wider text-ink-faint">
                              Final Confidence
                            </div>
                          </div>
                        </div>

                        <div className="p-4">
                          <div className="mb-2 flex items-center justify-between text-[10px] uppercase tracking-wider text-ink-faint">
                            <span>Confidence score</span>
                            <span className="font-mono">
                              {percentage.toFixed(1)}%
                            </span>
                          </div>

                          <div className="h-2 overflow-hidden rounded-full bg-panel-raised">
                            <div
                              className={`h-full rounded-full transition-all ${{
                                high: 'bg-signal',
                                review: 'bg-amber-400',
                                low: 'bg-alert',
                              }[confidenceLevel.tone]}`}
                              style={{ width: `${percentage}%` }}
                            />
                          </div>
                        </div>

                        <div className="flex flex-col gap-2 border-t border-hairline-soft px-4 py-3 text-[11px] text-ink-faint sm:flex-row sm:items-center sm:justify-between">
                          <span>
                            Validation:{' '}
                            {p.validation.replaceAll('_', ' ')}
                          </span>

                          <span className="font-mono">
                            bbox [{p.bbox.join(', ')}]
                          </span>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </Panel>
            )}
          </>
        )}
      </div>
    </div>
  )
}