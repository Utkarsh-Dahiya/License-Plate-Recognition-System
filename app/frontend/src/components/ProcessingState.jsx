import { Loader2, ImageIcon, Timer } from 'lucide-react'

/**
 * Loading state for an in-flight detection request.
 *
 * The backend exposes a single synchronous request/response endpoint — it
 * does not stream stages — so this deliberately does NOT fake a percentage or
 * an invented pipeline. It shows only what is genuinely known: that the image
 * is being analysed and how long it has been running.
 */
export default function ProcessingState({
  elapsedSeconds = 0,
  fileName,
  warmingUp = false,
}) {
  const seconds = Math.max(0, elapsedSeconds)

  return (
    <div
      role="status"
      aria-live="polite"
      className="flex flex-col items-center justify-center rounded-lg border border-hairline bg-panel p-10 text-center"
    >
      <div className="relative mb-5 flex h-16 w-16 items-center justify-center rounded-full border border-signal/30 bg-signal/5">
        <Loader2 size={28} className="animate-spin text-signal" aria-hidden="true" />
      </div>

      <h3 className="mb-1 font-display text-base font-bold text-ink">
        Analysis in progress
      </h3>
      <p className="mb-6 max-w-sm text-[12px] text-ink-dim">
        The image is being analysed by the plate detector and OCR engine. This screen updates
        automatically when the result is ready.
      </p>

      <div className="w-full max-w-sm space-y-2.5 text-left">
        {fileName && (
          <div className="flex items-center gap-3 rounded-md border border-hairline-soft bg-panel-raised/50 px-3 py-2 text-[12px]">
            <ImageIcon size={14} className="shrink-0 text-ink-faint" aria-hidden="true" />
            <span className="min-w-0 flex-1 truncate font-mono text-ink-dim">{fileName}</span>
          </div>
        )}

        <div className="flex items-center gap-3 rounded-md border border-signal/20 bg-signal/5 px-3 py-2 text-[12px]">
          <Timer size={14} className="shrink-0 text-signal" aria-hidden="true" />
          <span className="flex-1 text-ink-dim">Elapsed</span>
          <span className="font-mono font-semibold text-signal">
            {seconds < 10 ? seconds.toFixed(1) : Math.round(seconds)}s
          </span>
        </div>

        {warmingUp && (
          <p className="rounded-md border border-amber-500/20 bg-amber-500/5 px-3 py-2 text-[11px] leading-relaxed text-amber-300">
            The detection models are still loading on first use, so this first run takes longer
            than usual. Keep this page open.
          </p>
        )}

        <p className="px-1 pt-1 text-[11px] leading-relaxed text-ink-faint">
          Only one analysis runs at a time — the Detect button stays disabled until this one
          finishes.
        </p>
      </div>
    </div>
  )
}
