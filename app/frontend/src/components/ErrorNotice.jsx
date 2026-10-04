import { AlertCircle, RotateCw, X, WifiOff, Clock } from 'lucide-react'
import { toUserError } from '../lib/errors'

const KIND_ICON = {
  network: WifiOff,
  timeout: Clock,
}

/**
 * Single, consistent presentation for every failure in Image Analysis.
 *
 * Takes the raw error (ApiError from lib/api or a client-side userError),
 * runs it through toUserError(), and renders a title + plain-language
 * explanation plus recovery actions. Raw exception text and stack traces
 * never reach the DOM.
 */
export default function ErrorNotice({ error, onRetry, onDismiss, retryLabel = 'Retry detection' }) {
  if (!error) return null

  const info = toUserError(error)
  const Icon = KIND_ICON[info.kind] || AlertCircle
  const canRetry = Boolean(onRetry) && info.retryable !== false

  return (
    <div
      role="alert"
      aria-live="assertive"
      className="flex items-start gap-3 rounded-lg border border-alert/30 bg-alert/10 p-4"
    >
      <Icon size={16} className="mt-0.5 shrink-0 text-alert" aria-hidden="true" />

      <div className="min-w-0 flex-1">
        <p className="text-[13px] font-semibold text-alert">{info.title}</p>
        <p className="mt-1 text-[12px] leading-relaxed text-ink-dim">{info.message}</p>

        {(canRetry || onDismiss) && (
          <div className="mt-3 flex flex-wrap items-center gap-2">
            {canRetry && (
              <button
                type="button"
                onClick={onRetry}
                className="flex items-center gap-1.5 rounded-md border border-alert/40 px-3 py-1.5 text-[12px] font-medium text-alert transition-colors hover:bg-alert/15"
              >
                <RotateCw size={12} aria-hidden="true" />
                {retryLabel}
              </button>
            )}
            {onDismiss && (
              <button
                type="button"
                onClick={onDismiss}
                className="flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-[12px] font-medium text-ink-dim transition-colors hover:bg-panel-raised hover:text-ink"
              >
                <X size={12} aria-hidden="true" />
                Dismiss
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}