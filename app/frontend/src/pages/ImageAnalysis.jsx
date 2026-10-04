import { useCallback, useRef, useState } from 'react'
import {
  Sliders,
  Play,
  Loader2,
  Download,
  ImageIcon,
  X,
  Sparkles,
} from 'lucide-react'
import Header from '../components/Header.jsx'
import { Panel } from '../components/Panel.jsx'
import SamplePicker from '../components/SamplePicker.jsx'
import UploadDropzone from '../components/UploadDropzone.jsx'
import InteractiveCanvas from '../components/InteractiveCanvas.jsx'
import PlateResultCard from '../components/PlateResultCard.jsx'
import ProcessingState from '../components/ProcessingState.jsx'
import EmptyState from '../components/EmptyState.jsx'
import ErrorNotice from '../components/ErrorNotice.jsx'
import { useDetection } from '../hooks/useDetection'

function exportResults(result, format) {
  if (!result || !result.plates) return

  const rows = result.plates.map((p) => ({
    plate_id: p.plate_id,
    ocr_text: p.ocr_text || '',
    yolo_confidence: p.yolo_confidence,
    ocr_confidence: p.ocr_confidence,
    final_confidence: p.final_confidence,
    status: p.status,
    state_code: p.state_code || '',
    state_name: p.state_name || '',
    strict_format: p.strict_format ?? '',
    validation_score: p.validation_score ?? '',
    bbox: (p.bbox || []).join(' '),
    processing_time_seconds: result.processing_time_seconds,
    timestamp: new Date().toISOString(),
  }))

  const filename = `license-plate-detection-${Date.now()}.${format}`

  if (format === 'json') {
    const blob = new Blob(
      [
        JSON.stringify(
          {
            platform: 'License Plate Detection & OCR System',
            generated_at: new Date().toISOString(),
            plates_detected: result.plates_detected,
            processing_time_seconds: result.processing_time_seconds,
            image_dimensions: result.image,
            plates: rows,
          },
          null,
          2
        ),
      ],
      { type: 'application/json' }
    )
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
    return
  }

  const headers = [
    'plate_id',
    'ocr_text',
    'state_code',
    'state_name',
    'yolo_confidence',
    'ocr_confidence',
    'final_confidence',
    'status',
    'strict_format',
    'validation_score',
    'bbox',
    'processing_time_seconds',
    'timestamp',
  ]
  const escapeCsv = (v) => {
    const s = String(v ?? '')
    return /[",\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s
  }
  const csvContent = [
    headers.join(','),
    ...rows.map((r) => headers.map((h) => escapeCsv(r[h])).join(',')),
  ].join('\n')

  const blob = new Blob([csvContent], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

export default function ImageAnalysis() {
  const {
    file,
    previewUrl,
    fileMeta,
    confThreshold,
    setConfThreshold,
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
    isProcessing,
    selectFile,
    selectSample,
    runDetection,
    reset,
  } = useDetection()

  const replaceInputRef = useRef(null)

  // The catalogue is real state, so the "try a sample" action in the empty
  // state can act on an actual sample object. No document.querySelector.
  const [samples, setSamples] = useState([])
  const handleSamplesLoaded = useCallback((list) => setSamples(list), [])

  const canDetect = Boolean(file) && !isProcessing && sampleStatus !== 'loading'
  const hasResult = Boolean(result)
  const plateCount = result?.plates_detected ?? 0
  const plates = result?.plates ?? []

  /** Pick the first available curated sample (used by the empty state). */
  const pickFirstSample = useCallback(() => {
    const candidate = samples.find((s) => s.available !== false) || samples[0]
    if (candidate) selectSample(candidate)
  }, [samples, selectSample])

  /*
   * Note: there is deliberately no global Enter/Escape shortcut here. It used
   * to fire detection from a window-level keydown listener, which also fired
   * when Enter was pressed on any focused control and could submit twice.
   * The Detect button is a real, keyboard-operable control instead.
   */

  return (
    <div className="pb-16">
      <Header
        title="Live Detection Studio"
        description="Inspect vehicle images using the multi-pass YOLO localizer and adaptive OCR cascade."
      />

      <div className="mx-auto max-w-7xl space-y-6 px-4 py-6 sm:px-6">
        {/* Curated Sample Gallery */}
        <SamplePicker
          selectedSampleId={selectedSampleId}
          onSelect={selectSample}
          disabled={isProcessing}
          loadingSampleId={sampleStatus === 'loading' ? selectedSampleId : null}
          onSamplesLoaded={handleSamplesLoaded}
        />

        {/* Upload & Configuration Controls */}
        {!previewUrl ? (
          <UploadDropzone onFile={selectFile} disabled={isProcessing} busy={isProcessing} />
        ) : (
          /* Active Image Toolbar & Parameters */
          <div className="rounded-lg border border-hairline bg-panel p-4">
            <div className="flex flex-wrap items-center justify-between gap-4">
              {/* File Info + thumbnail preview */}
              <div className="flex min-w-0 items-center gap-3">
                <div className="h-12 w-16 shrink-0 overflow-hidden rounded border border-hairline bg-black/40">
                  <img
                    src={previewUrl}
                    alt={`Preview of ${fileMeta?.name || 'the selected image'}`}
                    className="h-full w-full object-cover"
                  />
                </div>
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="truncate text-[13px] font-semibold text-ink">
                      {fileMeta?.name}
                    </span>
                    {fileMeta?.sampleTitle && (
                      <span className="rounded border border-signal/20 bg-signal/10 px-1.5 py-0.2 font-mono text-[9px] font-semibold text-signal">
                        {fileMeta.sampleTitle}
                      </span>
                    )}
                  </div>
                  <div className="mt-0.5 font-mono text-[11px] text-ink-faint">
                    {fileMeta?.size} · {fileMeta?.type}
                    {fileMeta?.width ? ` · ${fileMeta.width}×${fileMeta.height}px` : ''}
                  </div>
                </div>
              </div>

              {/* Confidence Threshold Slider Control */}
              <div className="flex items-center gap-3 rounded-md border border-hairline-soft bg-panel-raised/70 px-3 py-1.5">
                <Sliders size={13} className="text-ink-dim" aria-hidden="true" />
                <label htmlFor="conf-threshold" className="text-[11px] font-medium text-ink-dim">
                  YOLO Confidence:
                </label>
                <input
                  id="conf-threshold"
                  type="range"
                  min="0.05"
                  max="0.95"
                  step="0.05"
                  value={confThreshold}
                  onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
                  disabled={isProcessing}
                  aria-valuetext={`${(confThreshold * 100).toFixed(0)} percent`}
                  className="w-24 cursor-pointer accent-signal"
                />
                <span className="w-10 text-right font-mono text-[11px] font-bold text-signal">
                  {(confThreshold * 100).toFixed(0)}%
                </span>
              </div>

              {/* Action Buttons */}
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={() => runDetection()}
                  disabled={!canDetect}
                  aria-busy={isProcessing}
                  className="flex items-center gap-2 rounded-md bg-signal px-4 py-2 text-[13px] font-semibold text-[#062015] shadow-sm transition-colors hover:bg-signal/90 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {isProcessing ? (
                    <>
                      <Loader2 size={15} className="animate-spin" aria-hidden="true" />
                      <span>Analysing…</span>
                    </>
                  ) : (
                    <>
                      <Play size={14} fill="currentColor" aria-hidden="true" />
                      <span>{hasResult ? 'Re-run Detection' : 'Detect Plates'}</span>
                    </>
                  )}
                </button>

                <button
                  type="button"
                  onClick={() => replaceInputRef.current?.click()}
                  disabled={isProcessing}
                  className="flex items-center gap-1.5 rounded-md border border-hairline px-3 py-2 text-[12px] font-medium text-ink-dim transition-colors hover:bg-panel-raised hover:text-ink disabled:cursor-not-allowed disabled:opacity-50"
                  title="Choose a different image"
                >
                  <ImageIcon size={13} aria-hidden="true" />
                  <span>Change</span>
                </button>

                <button
                  type="button"
                  onClick={reset}
                  disabled={isProcessing}
                  className="flex items-center gap-1.5 rounded-md border border-hairline px-3 py-2 text-[12px] font-medium text-ink-dim transition-colors hover:bg-panel-raised hover:text-ink disabled:cursor-not-allowed disabled:opacity-50"
                  title="Clear the image and return to the upload screen"
                >
                  <X size={13} aria-hidden="true" />
                  <span>Clear</span>
                </button>
              </div>
            </div>

            {/* Hidden picker reused by the "Change" action */}
            <input
              ref={replaceInputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              className="sr-only"
              aria-hidden="true"
              tabIndex={-1}
              onChange={(e) => {
                const picked = e.target.files?.[0]
                e.target.value = ''
                if (picked) selectFile(picked)
              }}
            />

            {/* Model Warmup notice */}
            {warmingUp && !hasResult && !isProcessing && (
              <div className="mt-3 flex items-center gap-2 rounded-md border border-amber-500/20 bg-amber-500/5 px-3 py-1.5 text-[11px] text-amber-300">
                <Loader2 size={12} className="shrink-0 animate-spin" aria-hidden="true" />
                <span>
                  Neural models are initializing in the background. The first detection may take a
                  few moments.
                </span>
              </div>
            )}
          </div>
        )}

        {/* Error Alert. Retry only makes sense when there is an image to
            re-run; sampleStatus is a string ('idle'/'loading'), so it must
            be compared explicitly rather than tested for truthiness. */}
        {error && (
          <ErrorNotice
            error={error}
            onRetry={file && sampleStatus !== 'loading' ? () => runDetection() : null}
            onDismiss={reset}
          />
        )}

        {/* Results Workspace: Dual Column Canvas + Intelligence.
            The canvas stays on screen during a run so the selected image is
            never lost; the progress panel takes over the results column. */}
        {previewUrl && (
          <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-12">
            {/* Viewport Canvas (Left 7 cols) */}
            <div className="space-y-4 lg:col-span-7">
              <InteractiveCanvas
                src={previewUrl}
                annotatedSrc={result?.annotated_image_base64}
                imageMeta={result?.image}
                plates={plates}
                hoveredPlateId={hoveredPlateId}
                selectedPlateId={selectedPlateId}
                onHoverPlate={setHoveredPlateId}
                onSelectPlate={setSelectedPlateId}
                fileName={fileMeta?.name}
              />
            </div>

            {/* Plate Results & Intelligence Panel (Right 5 cols) */}
            <div className="space-y-4 lg:col-span-5">
              <Panel
                title="Recognition Intelligence"
                subtitle={
                  hasResult
                    ? `${plateCount} plate${plateCount === 1 ? '' : 's'} localized in ${result.processing_time_seconds}s`
                    : isProcessing
                      ? 'Analysis running'
                      : 'Awaiting inference'
                }
              >
                {isProcessing ? (
                  <ProcessingState
                    elapsedSeconds={elapsedSeconds}
                    fileName={fileMeta?.name}
                    warmingUp={warmingUp}
                  />
                ) : !hasResult ? (
                  <div className="py-10 text-center">
                    <Sparkles size={20} className="mx-auto mb-3 text-ink-faint" aria-hidden="true" />
                    <p className="text-[12px] leading-relaxed text-ink-faint">
                      Press <span className="font-mono text-ink-dim">Detect Plates</span> to run
                      plate localization and OCR on this image.
                    </p>
                  </div>
                ) : plateCount === 0 ? (
                  <EmptyState
                    imageName={fileMeta?.name}
                    processingTimeSeconds={result.processing_time_seconds}
                    confThreshold={confThreshold}
                    onPickSample={pickFirstSample}
                    onLowerThresholdAndRun={
                      confThreshold > 0.05
                        ? () => {
                            setConfThreshold(0.05)
                            runDetection(0.05)
                          }
                        : null
                    }
                    onReset={reset}
                  />
                ) : (
                  <div className="space-y-4">
                    {/* Export Actions Toolbar */}
                    <div className="flex flex-wrap items-center justify-between gap-2 border-b border-hairline-soft pb-3 text-[11px]">
                      <span className="font-mono text-ink-dim">
                        Detector threshold &ge; {(confThreshold * 100).toFixed(0)}%
                      </span>
                      <div className="flex items-center gap-1.5">
                        <button
                          type="button"
                          onClick={() => exportResults(result, 'csv')}
                          aria-label="Export detected plates as CSV"
                          className="flex items-center gap-1 rounded border border-hairline bg-panel px-2.5 py-1 text-ink-dim transition-colors hover:bg-panel-raised hover:text-ink"
                          title="Export localized plates as CSV"
                        >
                          <Download size={11} aria-hidden="true" /> CSV
                        </button>
                        <button
                          type="button"
                          onClick={() => exportResults(result, 'json')}
                          aria-label="Export the detection payload as JSON"
                          className="flex items-center gap-1 rounded border border-hairline bg-panel px-2.5 py-1 text-ink-dim transition-colors hover:bg-panel-raised hover:text-ink"
                          title="Export intelligence payload as JSON"
                        >
                          <Download size={11} aria-hidden="true" /> JSON
                        </button>
                      </div>
                    </div>

                    {/* Detected Plates List */}
                    <div className="space-y-3">
                      {plates.map((plate) => (
                        <PlateResultCard
                          key={plate.plate_id}
                          plate={plate}
                          imageSrc={previewUrl}
                          imageMeta={result.image}
                          isSelected={selectedPlateId === plate.plate_id}
                          isHovered={hoveredPlateId === plate.plate_id}
                          onHover={setHoveredPlateId}
                          onSelect={setSelectedPlateId}
                        />
                      ))}
                    </div>
                  </div>
                )}
              </Panel>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
