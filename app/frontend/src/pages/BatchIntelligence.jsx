import { useEffect, useState } from 'react'
import { Search } from 'lucide-react'
import Header from '../components/Header.jsx'
import { Panel } from '../components/Panel.jsx'
import { StatRail } from '../components/StatRail.jsx'
import StatusBadge from '../components/StatusBadge.jsx'
import { api } from '../lib/api'

const STATUS_FILTERS = ['ALL', 'SUCCESS', 'OCR_FAILED', 'NO_PLATE']
const PAGE_SIZE = 20

export default function BatchIntelligence() {
  const [analytics, setAnalytics] = useState(null)
  const [rows, setRows] = useState({ total: 0, results: [] })
  const [status, setStatus] = useState('ALL')
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.batchAnalytics().then(setAnalytics).catch(() => setAnalytics(null))
  }, [])

  useEffect(() => {
    setLoading(true)
    const params = { page, page_size: PAGE_SIZE }
    if (status !== 'ALL') params.status = status
    if (search) params.search = search
    api
      .batchResults(params)
      .then(setRows)
      .finally(() => setLoading(false))
  }, [status, search, page])

  const stats = analytics
    ? [
        { label: 'Total images', value: analytics.dataset.total_images },
        { label: 'Total detections', value: analytics.dataset.total_plates },
        { label: 'OCR text extracted', value: analytics.ocr.success_rate.toFixed(1), unit: '%', tone: 'signal' },
        { label: 'OCR failures', value: analytics.ocr.failed },
        { label: 'Avg YOLO confidence', value: analytics.detection.mean_yolo_confidence.toFixed(1), unit: '%' },
        { label: 'Multi-plate images', value: analytics.detection.multi_plate_images },
      ]
    : []

  const totalPages = Math.max(1, Math.ceil(rows.total / PAGE_SIZE))

  return (
    <div>
      <Header
        title="Batch Intelligence"
        description="Results from the 658-image batch run — read from the existing CSV, never recomputed on load. SUCCESS means some text was extracted, not that the plate string is verified."
      />

      <div className="space-y-6 px-8 py-6">
        {analytics ? (
          <StatRail stats={stats} />
        ) : (
          <div className="h-24 animate-pulse rounded-lg border border-hairline bg-panel" />
        )}

        <Panel bodyClassName="p-0">
          <div className="flex flex-wrap items-center gap-3 border-b border-hairline-soft px-5 py-3">
            <div className="flex items-center gap-2 rounded-md border border-hairline bg-panel-raised px-2.5 py-1.5">
              <Search size={13} className="text-ink-faint" />
              <input
                value={search}
                onChange={(e) => {
                  setPage(1)
                  setSearch(e.target.value)
                }}
                placeholder="Search image or plate"
                className="w-40 bg-transparent text-[12px] text-ink placeholder:text-ink-faint focus:outline-none"
              />
            </div>
            <div className="flex gap-1">
              {STATUS_FILTERS.map((s) => (
                <button
                  key={s}
                  onClick={() => {
                    setPage(1)
                    setStatus(s)
                  }}
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
            <span className="ml-auto font-mono text-[11px] text-ink-faint">
              {rows.total} result{rows.total === 1 ? '' : 's'}
            </span>
          </div>

          <table className="w-full text-left text-[12px]">
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
              {rows.results.map((r, i) => (
                <tr key={i} className="border-t border-hairline-soft">
                  <td className="px-5 py-2 text-ink">{r.image}</td>
                  <td className="px-5 py-2 text-ink-dim">{r.plates_detected}</td>
                  <td className="px-5 py-2 font-mono text-ink-dim">
                    {(parseFloat(r.best_yolo_confidence) * 100).toFixed(1)}%
                  </td>
                  <td className="px-5 py-2 font-mono text-ink">{r.plate_text || '—'}</td>
                  <td className="px-5 py-2 font-mono text-ink-dim">
                    {(parseFloat(r.ocr_confidence) * 100).toFixed(1)}%
                  </td>
                  <td className="px-5 py-2">
                    <StatusBadge status={r.status} />
                  </td>
                </tr>
              ))}
              {!loading && rows.results.length === 0 && (
                <tr>
                  <td colSpan={6} className="px-5 py-8 text-center text-ink-faint">
                    No results match this filter.
                  </td>
                </tr>
              )}
            </tbody>
          </table>

          <div className="flex items-center justify-between border-t border-hairline-soft px-5 py-3 text-[11px] text-ink-dim">
            <span>
              Page {page} of {totalPages}
            </span>
            <div className="flex gap-2">
              <button
                disabled={page <= 1}
                onClick={() => setPage((p) => p - 1)}
                className="rounded border border-hairline px-2 py-1 disabled:opacity-40"
              >
                Prev
              </button>
              <button
                disabled={page >= totalPages}
                onClick={() => setPage((p) => p + 1)}
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
