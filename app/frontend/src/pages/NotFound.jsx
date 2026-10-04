import { Link } from 'react-router-dom'
import { FileQuestion, ArrowLeft, ImagePlay } from 'lucide-react'
import Header from '../components/Header.jsx'

/**
 * Catch-all route for unknown paths. Renders inside the normal app layout
 * (sidebar included) so navigation stays available.
 */
export default function NotFound() {
  return (
    <div className="pb-16">
      <Header
        title="Page not found"
        description="The address you opened does not match any view in this console."
      />

      <div className="px-6 py-10 sm:px-8">
        <div className="mx-auto max-w-md rounded-lg border border-dashed border-hairline bg-panel px-8 py-12 text-center">
          <div className="mx-auto mb-5 flex h-14 w-14 items-center justify-center rounded-full border border-hairline bg-panel-raised text-ink-faint">
            <FileQuestion size={24} aria-hidden="true" />
          </div>

          <p className="font-mono text-3xl font-bold tracking-tight text-ink">404</p>
          <h2 className="mt-2 font-display text-base font-bold text-ink">
            We could not find that page
          </h2>
          <p className="mt-2 text-[13px] leading-relaxed text-ink-dim">
            The link may be out of date or mistyped. Head back to the overview, or start a
            new image analysis.
          </p>

          <div className="mt-6 flex flex-wrap items-center justify-center gap-2">
            <Link
              to="/"
              className="flex items-center gap-1.5 rounded-md bg-signal px-4 py-2 text-[13px] font-semibold text-[#062015] transition-colors hover:bg-signal/90"
            >
              <ArrowLeft size={13} aria-hidden="true" />
              Back to Overview
            </Link>
            <Link
              to="/image-analysis"
              className="flex items-center gap-1.5 rounded-md border border-hairline px-4 py-2 text-[13px] font-medium text-ink-dim transition-colors hover:bg-panel-raised hover:text-ink"
            >
              <ImagePlay size={13} aria-hidden="true" />
              Image Analysis
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}