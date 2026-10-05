import { useRef, useState } from 'react'
import { Loader2, Search } from 'lucide-react'
import Header from '../components/Header.jsx'
import { Panel } from '../components/Panel.jsx'
import { StatRail } from '../components/StatRail.jsx'
import StatusBadge from '../components/StatusBadge.jsx'
import ErrorNotice from '../components/ErrorNotice.jsx'
import { api } from '../lib/api'
import { useAsyncResource, useDebouncedValue } from '../hooks/useAsyncResource'

const STATUS_FILTERS = ['ALL', 'SUCCESS', 'OCR_FAILED', 'NO_PLATE']
const PAGE_SIZE = 20
// The box updates instantly, but the API only sees it once it settles, so
// typing never fires a request per keystroke.
const SEARCH_DEBOUNCE_MS = 300

/** '85.3' for a rate, '—' when the field is missing or malformed. */
function formatDecimal(value) {
  const n = Number(value)
  return Number.isFinite(n) ? n.toFixed(1) : '—'
}

/** Same guard for confidences stored as 0-1 ratios. */
function formatPercent(value) {
  const n = Number(value)
  return Number.isFinite(n) ? `${(n * 100).toFixed(1)}%` : '—'
}

export default function BatchIntelligence() {
  const [status, setStatus] = useState('ALL')
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(1)

  const settledSearch = useDebouncedValue(search, SEARCH_DEBOUNCE_MS)

  // Reset the page in the same render that sees a settled search/status
  // change, so the query below fires once with { new filter, page 1 } instead
  // of once for the page reset and again for the filter.
  const lastFilterRef = useRef({ search: settledSearch, status })
  if (
    lastFilterRef.current.search !== settledSearch ||
    lastFilterRef.current.status !== status
  ) {
    lastFilterRef.current = { search: settledSearch, status }
    if (page !== 1) setPage(1)
  }

  // Both resources carry loading/error/retry and drop out-of-order replies,
  // so a slow page 1 can never overwrite a newer filtered response.
  const analytics = useAsyncResource(() => api.batchAnalytics(), [])

  const rows = useAsyncResource(
    () => {
      const params = { page, page_size: PAGE_SIZE }
      if (status !== 'ALL') params.status = status
      const needle = settledSearch.trim()
      if (needle) params.search = needle
      return api.batchResults(params)
    },
    [status, settledSearch, page],
    { initialData: { total: 0, results: [] } },
  )

  const results = rows.data.results ?? []

  const dash = analytics.data
  const stats = dash
    ? [
        { label: 'Total images', value: dash.dataset?.total_images ?? '—' },
        { label: 'Total detections', value: dash.dataset?.total_plates ?? '—' },
        {
          label: 'OCR text extracted',
          value: formatDecimal(dash.ocr?.success_rate),
          unit: '%',
          tone: 'signal',
        },
        { label: 'OCR failures', value: dash.ocr?.failed ?? '—' },
        {
          label: 'Avg YOLO confidence',
          value: formatDecimal(dash.detection?.mean_yolo_confidence),
          unit: '%',
        },
        {
          label: 'Multi-plate images',
          value: dash.detection?.multi_plate_images ?? '—',
        },
      ]
    : []

  const totalResults = rows.data.total ?? 0
  const totalPages = Math.max(1, Math.ceil(totalResults / PAGE_SIZE))

  return (
    <div>
      <Header
        title="Batch Intelligence"
        description="Results from the 658-image batch run — read from the existing CSV, never recomputed on load. SUCCESS means some text was extracted, not that the plate string is verified."
      />

      <div className="space-y-6 px-8 py-6">
        {/* First load is a labelled skeleton rather than a silent empty box,
            and a failed analytics fetch can be retried instead of hanging. */}
        {analytics.error && (
          <ErrorNotice
            error={analytics.error}
            onRetry={analytics.retry}
            retryLabel="Reload analytics"
          />
        )}

        {analytics.data ? (
          <StatRail stats={stats} />
        ) : !analytics.error ? (
          <div
            role="status"
            className="h-24 animate-pulse rounded-lg border border-hairline bg-panel"
          >
            <span className="sr-only">Loading batch analytics…</span>
          </div>
        ) : null}

        <Panel bodyClassName="p-0">
          <div className="flex flex-wrap items-center gap-3 border-b border-hairline-soft px-5 py-3">
            <div className="flex items-center gap-2 rounded-md border border-hairline bg-panel-raised px-2.5 py-1.5">
              <Search size={13} className="text-ink-faint" aria-hidden="true" />
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                aria-label="Search image or plate"
                placeholder="Search image or plate"
                className="w-40 bg-transparent text-[12px] text-ink placeholder:text-ink-faint focus:outline-none"
              />
            </div>
            <div
              className="flex gap-1"
              role="group"
              aria-label="Filter results by status"
            >
              {STATUS_FILTERS.map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => setStatus(s)}
                  aria-pressed={status === s}
                  className={`rounded-md px-2.5 py-1 text-[11px] font-medium transition-colors ${
                    status === s
                      ? 'bg-panel-raised text-ink'
                      : 'text-ink-dim hover:text-ink'
                  }`}
                >
                  {s.replaceAll('_', ' ')}
                </button>
              ))}
            </div>
            <span
              role="status"
              className="ml-auto font-mono text-[11px] text-ink-faint"
            >
              {rows.loading && totalResults === 0
                ? 'Loading…'
                : `${totalResults} result${totalResults === 1 ? '' : 's'}`}
            </span>
          </div>

          {/* The notice sits above the table so a failed page load is never
              mistaken for an empty one, and navigation stays available to
              recover by moving to another page. */}
          {rows.error && (
            <div className="px-5 py-4">
              <ErrorNotice
                error={rows.error}
                onRetry={rows.retry}
                retryLabel="Reload results"
              />
            </div>
          )}

          {/* Scrollable region keeps all six columns readable at 390px. */}
          <div
            className="overflow-x-auto"
            role="region"
            aria-label="Batch results"
            aria-busy={rows.loading}
            tabIndex={0}
          >
            <table className="w-full min-w-[720px] text-left text-[12px]">
              <thead>
                <tr className="text-ink-faint">
                  <th className="px-5 py-2 font-medium">Image</th>
                  <th className="px-5 py-2 font-medium">Plates</th>
                  <th className="px-5 py-2 font-medium">YOLO Confidence</th>
                  <th className="px-5 py-2 font-medium">OCR Result</th>
                  <th className="px-5 py-2 font-medium">OCR Confidence</th>
                  <th className="px-5 py-2 font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {rows.loading && results.length === 0 && (
                  <tr>
                    <td colSpan={6} className="px-5 py-8 text-center">
                      <span
                        role="status"
                        className="inline-flex items-center gap-2 text-ink-faint"
                      >
                        <Loader2 size={13} className="animate-spin" aria-hidden="true" />
                        Loading results…
                      </span>
                    </td>
                  </tr>
                )}

                {results.map((r, i) => (
                  <tr
                    key={`${r.image ?? 'row'}-${i}`}
                    className="border-t border-hairline-soft"
                  >
                    <td className="px-5 py-2 text-ink">{r.image}</td>
                    <td className="px-5 py-2 text-ink-dim">{r.plates_detected}</td>
                    <td className="px-5 py-2 font-mono text-ink-dim">
                      {formatPercent(r.best_yolo_confidence)}
                    </td>
                    <td className="px-5 py-2 font-mono text-ink">{r.plate_text || '—'}</td>
                    <td className="px-5 py-2 font-mono text-ink-dim">
                      {formatPercent(r.ocr_confidence)}
                    </td>
                    <td className="px-5 py-2">
                      <StatusBadge status={r.status} />
                    </td>
                  </tr>
                ))}

                {/* Empty is distinct from loading and from an error, which the
                    notice above already covers. */}
                {!rows.loading && !rows.error && results.length === 0 && (
                  <tr>
                    <td colSpan={6} className="px-5 py-8 text-center text-ink-faint">
                      <p role="status">
                        {status === 'ALL' && !settledSearch.trim() && totalResults === 0
                          ? 'No batch results have been generated yet.'
                          : 'No results match this filter.'}
                      </p>

                      {(status !== 'ALL' || settledSearch.trim()) && (
                        <button
                          type="button"
                          onClick={() => {
                            setStatus('ALL')
                            setSearch('')
                          }}
                          className="mt-3 rounded border border-hairline px-2.5 py-1 text-[11px] text-ink-dim hover:text-ink"
                        >
                          Clear filters
                        </button>
                      )}
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <div className="flex items-center justify-between border-t border-hairline-soft px-5 py-3 text-[11px] text-ink-dim">
            <span role="status">
              Page {page} of {totalPages}
              {rows.loading && results.length > 0 ? ' · updating…' : ''}
            </span>
            <div className="flex gap-2">
              <button
                type="button"
                disabled={page <= 1 || rows.loading}
                onClick={() => setPage((p) => p - 1)}
                aria-label="Previous page"
                className="rounded border border-hairline px-2 py-1 disabled:opacity-40"
              >
                Prev
              </button>
              <button
                type="button"
                disabled={page >= totalPages || rows.loading}
                onClick={() => setPage((p) => p + 1)}
                aria-label="Next page"
                className="rounded border border-hairline px-2 py-1 disabled:opacity-40"
              >
                Next
              </button>
            </div>
          </div>
        </Panel>
      </div>
    </div>
  )
}
