import { useRef, useState } from 'react'
import { Layers, Download } from 'lucide-react'

export default function InteractiveCanvas({
  src,
  annotatedSrc,
  imageMeta,
  plates = [],
  hoveredPlateId,
  selectedPlateId,
  onHoverPlate,
  onSelectPlate,
  fileName,
}) {
  const [viewMode, setViewMode] = useState('interactive') // 'interactive' | 'annotated'
  const containerRef = useRef(null)

  const imgW = Number(imageMeta?.width) || 1280
  const imgH = Number(imageMeta?.height) || 720

  /**
   * The annotated frame is returned by the API as base64 JPEG
   * (annotated_image_base64). Decode it into a Blob so the browser writes a
   * real .jpg file rather than navigating to a data: URL.
   */
  function downloadAnnotated() {
    if (!annotatedSrc) return
    try {
      const binary = atob(annotatedSrc)
      const bytes = new Uint8Array(binary.length)
      for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i)
      const blob = new Blob([bytes], { type: 'image/jpeg' })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `${(fileName || 'image').replace(/\.[^.]+$/, '')}-annotated.jpg`
      link.click()
      URL.revokeObjectURL(url)
    } catch {
      // A malformed base64 payload should not break the page.
    }
  }

  return (
    <div className="relative w-full overflow-hidden rounded-lg border border-hairline bg-panel">
      {/* Header toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-hairline-soft bg-panel-raised/60 px-4 py-2">
        <div className="flex min-w-0 items-center gap-2">
          <Layers size={13} className="shrink-0 text-signal" aria-hidden="true" />
          <span className="text-[12px] font-medium text-ink">Detection Viewport</span>
          {plates.length > 0 && (
            <span className="rounded border border-hairline-soft bg-panel px-2 py-0.5 font-mono text-[10px] text-ink-dim">
              {plates.length} plate{plates.length === 1 ? '' : 's'} localized
            </span>
          )}
        </div>

        {annotatedSrc && (
          <div className="flex flex-wrap items-center gap-1.5">
            <div
              className="flex items-center gap-1 rounded-md border border-hairline-soft bg-panel p-0.5"
              role="group"
              aria-label="Image view mode"
            >
              <button
                type="button"
                onClick={() => setViewMode('interactive')}
                aria-pressed={viewMode === 'interactive'}
                className={`rounded px-2.5 py-1 text-[11px] font-medium transition-colors ${
                  viewMode === 'interactive'
                    ? 'bg-signal font-semibold text-[#062015] shadow-sm'
                    : 'text-ink-dim hover:text-ink'
                }`}
              >
                Interactive Overlay
              </button>
              <button
                type="button"
                onClick={() => setViewMode('annotated')}
                aria-pressed={viewMode === 'annotated'}
                className={`rounded px-2.5 py-1 text-[11px] font-medium transition-colors ${
                  viewMode === 'annotated'
                    ? 'bg-signal font-semibold text-[#062015] shadow-sm'
                    : 'text-ink-dim hover:text-ink'
                }`}
              >
                Model Annotated
              </button>
            </div>

            <button
              type="button"
              onClick={downloadAnnotated}
              aria-label="Download the annotated image"
              title="Download annotated image"
              className="flex items-center gap-1 rounded-md border border-hairline bg-panel px-2.5 py-1.5 text-[11px] font-medium text-ink-dim transition-colors hover:bg-panel-raised hover:text-ink"
            >
              <Download size={11} aria-hidden="true" />
              PNG
            </button>
          </div>
        )}
      </div>

      {/* Main Image & Overlay Canvas */}
      <div
        ref={containerRef}
        className="relative flex w-full items-center justify-center overflow-hidden bg-black/60"
        style={{ minHeight: '240px' }}
      >
        {viewMode === 'annotated' && annotatedSrc ? (
          <img
            src={`data:image/jpeg;base64,${annotatedSrc}`}
            alt="Processed image with detection boxes and recognised plate text drawn by the model"
            className="block h-auto max-h-[640px] w-auto max-w-full select-none object-contain"
          />
        ) : (
          /*
           * The wrapper is w-fit so it matches the RENDERED image box exactly.
           * A full-width wrapper plus object-contain letterboxes tall images,
           * which shifted every bounding box away from its plate.
           */
          <div className="relative w-fit max-w-full">
            {/* Base Image */}
            <img
              src={src}
              alt="Analysed image with detected plate regions highlighted"
              className="block h-auto max-h-[640px] w-auto max-w-full select-none"
            />

            {/* SVG Interactive Overlay using normalized viewBox */}
            {plates.length > 0 && (
              <svg
                viewBox={`0 0 ${imgW} ${imgH}`}
                preserveAspectRatio="none"
                className="pointer-events-none absolute inset-0 h-full w-full"
              >
                {plates.map((p) => {
                  if (!Array.isArray(p.bbox) || p.bbox.length < 4) return null
                  const [x1, y1, x2, y2] = p.bbox.map(Number)
                  const boxW = Math.max(1, x2 - x1)
                  const boxH = Math.max(1, y2 - y1)

                  const isHovered = hoveredPlateId === p.plate_id
                  const isSelected = selectedPlateId === p.plate_id
                  const hasText = Boolean(p.ocr_text)

                  let strokeColor = '#3ED98B' // signal green
                  let fillColor = 'rgba(62, 217, 139, 0.15)'
                  if (!hasText || p.status === 'OCR_FAILED') {
                    strokeColor = '#EF5B54' // alert red
                    fillColor = 'rgba(239, 91, 84, 0.15)'
                  } else if (Number(p.final_confidence || 0) < 0.8) {
                    strokeColor = '#F2A93B' // review yellow
                    fillColor = 'rgba(242, 169, 59, 0.15)'
                  }

                  if (isSelected) {
                    strokeColor = '#FFFFFF'
                    fillColor = 'rgba(255, 255, 255, 0.25)'
                  } else if (isHovered) {
                    fillColor = 'rgba(62, 217, 139, 0.35)'
                  }

                  return (
                    <g
                      key={p.plate_id}
                      className="pointer-events-auto cursor-pointer transition-all duration-150"
                      onMouseEnter={() => onHoverPlate?.(p.plate_id)}
                      onMouseLeave={() => onHoverPlate?.(null)}
                      onFocus={() => onHoverPlate?.(p.plate_id)}
                      onBlur={() => onHoverPlate?.(null)}
                      onClick={() => onSelectPlate?.(p.plate_id)}
                      tabIndex={0}
                      role="button"
                      aria-pressed={isSelected}
                      aria-label={`Detected plate ${p.plate_id}: ${p.ocr_text || 'Text unread'}`}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault()
                          onSelectPlate?.(p.plate_id)
                        }
                      }}
                    >
                      {/* Bounding box rectangle */}
                      <rect
                        x={x1}
                        y={y1}
                        width={boxW}
                        height={boxH}
                        fill={fillColor}
                        stroke={strokeColor}
                        strokeWidth={isSelected ? Math.max(3, imgW / 300) : isHovered ? Math.max(2.5, imgW / 360) : Math.max(1.8, imgW / 450)}
                        rx={Math.max(2, imgW / 400)}
                        className="transition-all"
                        filter={isSelected ? 'drop-shadow(0 0 8px rgba(255,255,255,0.7))' : isHovered ? 'drop-shadow(0 0 6px rgba(62,217,139,0.7))' : undefined}
                      />

                      {/* Tag badge with Plate number and OCR text */}
                      <g transform={`translate(${x1}, ${Math.max(0, y1 - Math.max(22, imgH / 28))})`}>
                        <rect
                          x={0}
                          y={0}
                          width={Math.max(70, Math.min(boxW * 1.2, (p.ocr_text?.length || 4) * 14 + 32))}
                          height={Math.max(20, imgH / 30)}
                          fill="#0D1219"
                          stroke={strokeColor}
                          strokeWidth={1}
                          rx={3}
                          opacity={0.95}
                        />
                        <text
                          x={6}
                          y={Math.max(14, imgH / 42)}
                          fill={strokeColor}
                          fontSize={Math.max(11, imgH / 48)}
                          fontFamily="monospace"
                          fontWeight="bold"
                        >
                          #{p.plate_id} {p.ocr_text || 'UNREAD'}
                        </text>
                      </g>
                    </g>
                  )
                })}
              </svg>
            )}
          </div>
        )}
      </div>

      {/* Viewport footer legend */}
      {plates.length > 0 && viewMode === 'interactive' && (
        <div className="flex flex-wrap items-center justify-between gap-2 border-t border-hairline-soft bg-panel-raised/40 px-4 py-2 text-[11px] text-ink-dim">
          <div className="flex flex-wrap items-center gap-3">
            <span className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-signal" aria-hidden="true" />
              High confidence (&ge;80%)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-review" aria-hidden="true" />
              Needs review (50-79%)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-alert" aria-hidden="true" />
              Low or unreadable
            </span>
          </div>
          <span className="font-mono text-ink-faint">
            Hover, focus or click any box to sync it with its result card
          </span>
        </div>
      )}
    </div>
  )
}
