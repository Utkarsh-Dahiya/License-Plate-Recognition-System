import { useCallback, useEffect, useState } from 'react'
import { Sparkles, ArrowRight, Loader2, RefreshCw, ImageOff, Check } from 'lucide-react'
import { api } from '../lib/api'
import { toUserError } from '../lib/errors'

/**
 * Curated sample gallery.
 *
 * Selection is driven entirely by `selectedSampleId` coming from the parent's
 * React state — this component never reads the DOM to discover what is
 * selected, and never mutates the selection itself. The fetched catalogue is
 * handed back to the parent via `onSamplesLoaded` so a "try a sample" action
 * can act on real data instead of querying the document.
 */
export default function SamplePicker({
  selectedSampleId,
  onSelect,
  disabled,
  loadingSampleId,
  onSamplesLoaded,
}) {
  const [samples, setSamples] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const load = useCallback(() => {
    let cancelled = false
    setLoading(true)
    setError(null)

    api
      .samples()
      .then((res) => {
        if (cancelled) return
        const list = Array.isArray(res?.samples) ? res.samples : []
        setSamples(list)
        onSamplesLoaded?.(list)
      })
      .catch((err) => {
        if (cancelled) return
        setSamples([])
        setError(toUserError(err))
        onSamplesLoaded?.([])
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [onSamplesLoaded])

  useEffect(() => load(), [load])

  if (loading) {
    return (
      <div className="rounded-lg border border-hairline bg-panel p-4">
        <div className="mb-3 flex items-center gap-2 text-[12px] text-ink-dim">
          <Sparkles size={14} className="text-signal" aria-hidden="true" />
          <span>Curated Sample Library</span>
        </div>
        <div
          className="grid grid-cols-2 gap-2.5 sm:grid-cols-3 lg:grid-cols-6"
          aria-busy="true"
          aria-label="Loading curated samples"
        >
          {[...Array(6)].map((_, i) => (
            <div
              key={i}
              className="h-28 animate-pulse rounded-md border border-hairline-soft bg-panel-raised/50"
            />
          ))}
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div
        role="status"
        className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-hairline bg-panel p-4"
      >
        <div className="min-w-0">
          <p className="text-[12px] font-semibold text-ink">Curated sample library unavailable</p>
          <p className="mt-0.5 text-[11px] leading-relaxed text-ink-faint">{error.message}</p>
        </div>
        <button
          type="button"
          onClick={load}
          className="flex shrink-0 items-center gap-1.5 rounded-md border border-hairline px-3 py-1.5 text-[11px] font-medium text-ink-dim transition-colors hover:bg-panel-raised hover:text-ink"
        >
          <RefreshCw size={12} aria-hidden="true" />
          Retry loading samples
        </button>
      </div>
    )
  }

  if (samples.length === 0) {
    return (
      <div className="rounded-lg border border-hairline bg-panel p-4">
        <p className="text-[12px] text-ink-dim">
          No curated samples are available on this server. Upload an image to run detection.
        </p>
      </div>
    )
  }

  return (
    <div className="rounded-lg border border-hairline bg-panel p-4">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <div className="flex items-center gap-2 text-ink text-[12px] font-semibold tracking-wide uppercase">
          <Sparkles size={14} className="text-signal" />
          <span>Curated Evaluation Gallery</span>
          <span className="rounded bg-panel-raised px-1.5 py-0.5 text-[10px] font-mono text-ink-faint">
            1-Click Test
          </span>
        </div>
        <span className="text-[11px] text-ink-faint">
          Select an Indian vehicle plate scenario to evaluate detection and OCR
        </span>
      </div>

      {/* radiogroup semantics: exactly one sample is the active selection. */}
      <div
        role="radiogroup"
        aria-label="Curated sample images"
        className="grid grid-cols-2 gap-2.5 sm:grid-cols-3 lg:grid-cols-6"
      >
        {samples.map((sample) => {
          const isSelected = selectedSampleId === sample.id
          const isAvailable = sample.available !== false
          const isLoadingThis = loadingSampleId === sample.id

          return (
            <button
              key={sample.id}
              type="button"
              role="radio"
              aria-checked={isSelected}
              disabled={!isAvailable || disabled || isLoadingThis}
              onClick={() => onSelect(sample)}
              title={
                isAvailable
                  ? `${sample.title}${sample.expected_plate && sample.expected_plate !== 'Multiple' ? ` — expected ${sample.expected_plate}` : ''}`
                  : 'Sample image is not present on the server'
              }
              className={`group relative flex flex-col justify-between overflow-hidden rounded-md border text-left transition-all ${
                !isAvailable
                  ? 'cursor-not-allowed border-hairline-soft bg-panel-raised/20 opacity-40'
                  : isSelected
                    ? 'border-signal bg-signal/5 ring-1 ring-signal shadow-lg shadow-signal/10'
                    : 'cursor-pointer border-hairline-soft bg-panel-raised hover:border-hairline hover:bg-panel-raised/80'
              }`}
            >
              {/* Thumbnail preview */}
              <div className="relative h-16 w-full overflow-hidden border-b border-hairline-soft bg-black/40">
                <img
                  src={api.sampleImageUrl(sample.id)}
                  alt={`${sample.title} sample vehicle`}
                  className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
                  loading="lazy"
                />
                <div className="absolute left-1 top-1 flex items-center gap-1 rounded bg-black/70 px-1.5 py-0.5 backdrop-blur-sm">
                  <span className="font-mono text-[9px] font-bold text-signal">
                    {sample.state_code || 'IND'}
                  </span>
                </div>
                {sample.difficulty && (
                  <div className="absolute bottom-1 right-1 rounded bg-black/70 px-1 py-0.5 text-[8px] font-mono uppercase tracking-wider text-ink-dim backdrop-blur-sm">
                    {sample.difficulty}
                  </div>
                )}

                {/* Loading / selected / unavailable affordance */}
                {isLoadingThis ? (
                  <span className="absolute inset-0 flex items-center justify-center bg-black/60">
                    <Loader2 size={18} className="animate-spin text-signal" aria-hidden="true" />
                    <span className="sr-only">Loading sample image</span>
                  </span>
                ) : isSelected ? (
                  <span className="absolute right-1 top-1 flex h-4 w-4 items-center justify-center rounded-full bg-signal text-[#062015]">
                    <Check size={11} strokeWidth={3} aria-hidden="true" />
                  </span>
                ) : !isAvailable ? (
                  <span className="absolute inset-0 flex items-center justify-center bg-black/60 text-ink-dim">
                    <ImageOff size={16} aria-hidden="true" />
                  </span>
                ) : null}
              </div>

              {/* Content info */}
              <div className="flex flex-1 flex-col justify-between p-2">
                <div>
                  <div
                    className={`text-[11px] font-medium leading-snug line-clamp-1 transition-colors ${
                      isSelected ? 'text-signal' : 'text-ink group-hover:text-signal'
                    }`}
                  >
                    {sample.title}
                  </div>
                  <div className="mt-0.5 truncate font-mono text-[10px] text-ink-faint">
                    {sample.state || sample.expected_plate}
                  </div>
                </div>

                <div className="mt-2 flex items-center justify-between border-t border-hairline-soft/60 pt-1.5">
                  <span className="text-[9px] font-mono uppercase text-ink-dim">
                    {sample.expected_plate}
                  </span>
                  {isAvailable && !isSelected && (
                    <span className="inline-flex items-center text-[10px] font-semibold text-signal opacity-0 transition-opacity group-hover:opacity-100">
                      Load
                      <ArrowRight size={10} className="ml-0.5" aria-hidden="true" />
                    </span>
                  )}
                </div>
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}
