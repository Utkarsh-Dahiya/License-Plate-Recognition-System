import { useState } from 'react'
import { Copy, Check, Download, Crosshair } from 'lucide-react'
import IndianPlateCard from './IndianPlateCard.jsx'
import StatusBadge, { confidenceBand } from './StatusBadge.jsx'

/**
 * Formats a confidence value only when the backend actually sent one.
 * Never substitutes a default — a missing metric shows as "not reported".
 */
function fmtPercent(val) {
  if (val === null || val === undefined || !Number.isFinite(Number(val))) return null
  return `${(Number(val) * 100).toFixed(1)}%`
}

/**
 * Format verdict, derived from the backend's STRICT validation layer.
 *
 * The legacy `validation` field (`indian_plate_score`) is a pure
 * character-shape heuristic: it takes no OCR confidence and scores any
 * plate-shaped blob highly (e.g. "EM01B21650" scores 90/100 even at 12.6%
 * OCR confidence). `strict_format` is the layer that knows real Indian
 * issuance rules — a real RTO state code plus a valid district/series — so
 * it is what the user-facing verdict is based on.
 *
 * Never falls back to the legacy field, and never invents a verdict: an
 * absent `strict_format` is reported as "Not reported".
 */
function formatVerdict(plate) {
  if (plate.strict_format === true) return 'Strict Indian format'
  if (plate.strict_format === false) return 'Not a strict Indian format'
  return null
}

/** Plain-language wording for the state the backend actually resolved. */
function stateLabel(plate) {
  if (plate.state_code && plate.state_name) return `${plate.state_code} · ${plate.state_name}`
  if (plate.state_code) return plate.state_code
  if (plate.state_name) return plate.state_name
  return null
}

function MetricRow({ label, value, tone = 'default' }) {
  return (
    <div className="flex items-center justify-between gap-2">
      <span className="text-ink-dim">{label}</span>
      <span
        className={`font-mono font-semibold ${
          value === null ? 'text-ink-faint' : tone === 'accent' ? 'text-signal' : 'text-ink'
        }`}
      >
        {value === null ? 'Not reported' : value}
      </span>
    </div>
  )
}

export default function PlateResultCard({
  plate,
  imageSrc,
  imageMeta,
  isSelected,
  isHovered,
  onHover,
  onSelect,
}) {
  const [copied, setCopied] = useState(false)

  if (!plate) return null

  const text = (plate.ocr_text || '').trim()
  const hasText = text.length > 0
  const status = plate.status || 'OCR_FAILED'
  const isStrict = plate.strict_format === true
  const band = hasText ? confidenceBand(plate.final_confidence) : 'OCR_FAILED'

  const yoloConf = fmtPercent(plate.yolo_confidence)
  const ocrConf = fmtPercent(plate.ocr_confidence)
  const finalConf = fmtPercent(plate.final_confidence)
  const validationScore =
    plate.validation_score === null || plate.validation_score === undefined
      ? null
      : `${plate.validation_score}/100`

  const hasBbox = Array.isArray(plate.bbox) && plate.bbox.length >= 4
  const [x1, y1, x2, y2] = hasBbox ? plate.bbox.map(Number) : [0, 0, 0, 0]
  const boxW = Math.max(1, x2 - x1)
  const boxH = Math.max(1, y2 - y1)

  async function handleCopy(e) {
    e.stopPropagation()
    if (!text) return
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      // Clipboard can be blocked by permissions/policy; the text stays visible.
    }
  }

  function handleDownloadCrop(e) {
    e.stopPropagation()
    if (!imageSrc || !hasBbox) return

    const imgEl = new Image()
    imgEl.onload = () => {
      try {
        const canvas = document.createElement('canvas')
        canvas.width = boxW
        canvas.height = boxH
        const ctx = canvas.getContext('2d')
        ctx.drawImage(imgEl, x1, y1, boxW, boxH, 0, 0, boxW, boxH)
        const link = document.createElement('a')
        link.href = canvas.toDataURL('image/png')
        link.download = `license-plate-detection-plate-${plate.plate_id}-${text || 'crop'}.png`
        link.click()
      } catch {
        // A tainted canvas (cross-origin source) cannot be exported.
      }
    }
    imgEl.src = imageSrc
  }

  // Client-side crop framing, expressed as fractions of the full image.
  const imgW = Number(imageMeta?.width) || 0
  const imgH = Number(imageMeta?.height) || 0
  const fw = Math.min(1, Math.max(0.01, boxW / (imgW || 1)))
  const fh = Math.min(1, Math.max(0.01, boxH / (imgH || 1)))

  // The card body itself is NOT a button: it used to be a role="button"
  // container wrapping real buttons, which is invalid and broke keyboard
  // and screen-reader interaction. Selection is now an explicit control.
  const cardBorder = isSelected
    ? 'border-signal bg-panel-raised ring-1 ring-signal'
    : isHovered
      ? 'border-hairline bg-panel-raised/80'
      : 'border-hairline-soft bg-panel hover:border-hairline'

  return (
    <article
      className={`rounded-lg border p-4 text-left transition-all ${cardBorder}`}
      onMouseEnter={() => onHover?.(plate.plate_id)}
      onMouseLeave={() => onHover?.(null)}
      aria-label={`Plate ${plate.plate_id}: ${hasText ? text : 'text not read'}`}
    >
      {/* Header bar */}
      <div className="mb-3 flex items-center justify-between gap-2 border-b border-hairline-soft pb-3">
        <div className="flex min-w-0 items-center gap-2">
          <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded border border-hairline bg-panel font-mono text-[10px] font-bold text-signal">
            {plate.plate_id}
          </span>
          <span className="truncate text-[12px] font-semibold text-ink">
            {hasText ? `Plate ${plate.plate_id}` : `Detection ${plate.plate_id}`}
          </span>
          <StatusBadge status={status} />
        </div>

        <div className="flex shrink-0 items-center gap-1.5">
          {hasText && (
            <button
              type="button"
              onClick={handleCopy}
              aria-label={`Copy plate text ${text}`}
              className="flex items-center gap-1 rounded border border-hairline bg-panel px-2 py-1 text-[10px] font-medium text-ink-dim transition-colors hover:bg-panel-raised hover:text-ink"
            >
              {copied ? (
                <Check size={11} className="text-signal" aria-hidden="true" />
              ) : (
                <Copy size={11} aria-hidden="true" />
              )}
              {copied ? 'Copied' : 'Copy'}
            </button>
          )}

          {hasBbox && (
            <button
              type="button"
              onClick={handleDownloadCrop}
              aria-label={`Download crop of plate ${plate.plate_id}`}
              className="flex items-center gap-1 rounded border border-hairline bg-panel px-2 py-1 text-[10px] font-medium text-ink-dim transition-colors hover:bg-panel-raised hover:text-ink"
            >
              <Download size={11} aria-hidden="true" />
              Crop
            </button>
          )}

          <button
            type="button"
            onClick={() => onSelect?.(isSelected ? null : plate.plate_id)}
            aria-pressed={isSelected}
            aria-label={
              isSelected
                ? `Clear highlight for plate ${plate.plate_id}`
                : `Highlight plate ${plate.plate_id} on the image`
            }
            title={isSelected ? 'Clear highlight' : 'Show on image'}
            className={`flex items-center gap-1 rounded border px-2 py-1 text-[10px] font-medium transition-colors ${
              isSelected
                ? 'border-signal bg-signal/15 text-signal'
                : 'border-hairline bg-panel text-ink-dim hover:bg-panel-raised hover:text-ink'
            }`}
          >
            <Crosshair size={11} aria-hidden="true" />
            {isSelected ? 'Showing' : 'Locate'}
          </button>
        </div>
      </div>

      {/* Plate rendering */}
      <div className="mb-4">
        <IndianPlateCard plate={plate} isSelected={isSelected} />
      </div>

      {/* Registration detail. The row renders whenever the backend returned a
          format verdict (including a negative one) so a rejected read is
          visibly explained instead of silently hidden. */}
      {(plate.state_code || plate.state_name || !isStrict) && (
        <div className="mb-3 flex flex-wrap items-center gap-2 text-[11px]">
          {plate.state_code && (
            <span className="rounded border border-hairline bg-panel-raised px-2 py-0.5 font-mono font-semibold text-signal">
              {plate.state_code}
            </span>
          )}
          {plate.state_name && <span className="text-ink-dim">{plate.state_name}</span>}
          <span
            className={`rounded border px-2 py-0.5 text-[10px] ${
              isStrict
                ? 'border-signal/30 bg-signal/10 text-signal'
                : 'border-hairline bg-panel-raised text-ink-dim'
            }`}
          >
            {isStrict
              ? 'Strict Indian format'
              : 'Non-standard format'}
          </span>
        </div>
      )}

      {/* Crop preview & metrics */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div className="flex flex-col justify-between rounded-md border border-hairline-soft bg-black/40 p-1.5">
          <div
            className="relative w-full overflow-hidden rounded bg-black/30"
            style={{ aspectRatio: `${boxW} / ${boxH}` }}
          >
            {imageSrc && imgW > 0 && imgH > 0 && hasBbox && (
              <img
                src={imageSrc}
                alt={`Cropped region of plate ${plate.plate_id}${hasText ? ` reading ${text}` : ''}`}
                className="absolute max-w-none"
                style={{
                  width: `${100 / fw}%`,
                  left: `${(-100 * (x1 / imgW)) / fw}%`,
                  top: `${(-100 * (y1 / imgH)) / fh}%`,
                }}
              />
            )}
          </div>
          <div className="mt-1 text-center font-mono text-[9px] uppercase tracking-wider text-ink-faint">
            Bounding box {boxW}×{boxH}px
          </div>
        </div>

        <div className="flex flex-col justify-center space-y-1.5 rounded-md border border-hairline-soft bg-panel-raised/40 p-2.5 text-[11px]">
          <MetricRow label="Detection confidence" value={yoloConf} />
          <MetricRow label="OCR confidence" value={ocrConf} />
          <div className="border-t border-hairline-soft pt-1">
            <MetricRow label="Combined confidence" value={finalConf} tone="accent" />
          </div>
          {/* The raw heuristic score is still surfaced (it is a real API field) but
              labelled so it cannot be mistaken for trustworthiness: it is a
              character-shape score with no OCR-confidence input, and it is
              deliberately NOT what the Format check verdict is based on. */}
          <MetricRow label="Format shape score" value={validationScore} />
          <MetricRow label="Format check" value={formatVerdict(plate)} />
          <MetricRow label="State" value={stateLabel(plate)} />
          <div className="flex items-center justify-between gap-2">
            <span className="text-ink-dim">Quality band</span>
            <StatusBadge status={band} />
          </div>
        </div>
      </div>
    </article>
  )
}
