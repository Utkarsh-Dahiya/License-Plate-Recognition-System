import { Loader2 } from 'lucide-react'

/**
 * Suspense fallback for lazily-loaded routes.
 *
 * Route-level code splitting means a page's chunk is fetched on navigation,
 * so a brief placeholder is needed while it resolves. It deliberately reuses
 * the existing visual language — the same dashed border, vertical rhythm and
 * muted type as `ComingNext` / `Panel` — so no new visual identity is added.
 *
 * This is a truthful "loading the view" state: it never claims a percentage,
 * a pipeline stage, or any fabricated progress.
 */
export default function PageFallback({ label = 'Loading view' }) {
  return (
    <div role="status" aria-live="polite" aria-busy="true" className="px-6 py-6 sm:px-8">
      <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-hairline px-8 py-20 text-center">
        <Loader2
          size={22}
          className="mb-4 animate-spin text-ink-faint"
          strokeWidth={1.5}
          aria-hidden="true"
        />
        <p className="text-[13px] text-ink-dim">{label}…</p>
        <span className="sr-only">Please wait while this page loads.</span>
      </div>
    </div>
  )
}