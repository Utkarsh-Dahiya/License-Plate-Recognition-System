const STYLES = {
  SUCCESS: 'bg-signal-dim/30 text-signal border-signal-dim',
  HIGH_CONFIDENCE: 'bg-signal-dim/30 text-signal border-signal-dim',
  REVIEW: 'bg-review/10 text-review border-review/40',
  LOW_CONFIDENCE: 'bg-alert/10 text-alert border-alert/40',
  OCR_FAILED: 'bg-alert/10 text-alert border-alert/40',
  NO_PLATE: 'bg-alert/10 text-alert border-alert/40',
}

/** Match backend: >=0.80 HIGH, >=0.50 REVIEW, else LOW. */
export function confidenceBand(value) {
  const n = Number(value)
  if (!Number.isFinite(n)) return 'UNKNOWN'
  if (n >= 0.8) return 'HIGH_CONFIDENCE'
  if (n >= 0.5) return 'REVIEW'
  return 'LOW_CONFIDENCE'
}

export default function StatusBadge({ status }) {
  const style = STYLES[status] || 'bg-ink-faint/10 text-ink-dim border-ink-faint/30'
  return (
    <span
      className={`inline-flex items-center rounded border px-2 py-0.5 text-[11px] font-medium ${style}`}
    >
      {status?.replaceAll('_', ' ') || 'UNKNOWN'}
    </span>
  )
}
