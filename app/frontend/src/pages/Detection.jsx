import { useEffect, useState } from 'react'
import { ScanLine, Image as ImageIcon, Video, Search, Download } from 'lucide-react'
import Header from '../components/Header.jsx'
import { Panel } from '../components/Panel.jsx'
import ComingNext from '../components/ComingNext.jsx'
import StatusBadge from '../components/StatusBadge.jsx'
import { api } from '../lib/api'

const FILTERS = ['ALL', 'image', 'video']

function fmtTimestamp(ts) {
  return new Date(ts * 1000).toLocaleString()
}

/**
 * Live system statistics from /api/history/stats. Every number comes
 * from actually recorded runs; when nothing has been recorded the
 * endpoint returns nulls and this panel renders an explicit empty
 * state — never fabricated numbers.
 */
function LiveStats({ stats }) {
  if (!stats) return null

  if (!stats.total_runs) {
    return (
      <Panel title="System statistics">
        <p className="py-4 text-center text-[12px] text-ink-faint">
          No runs recorded yet — statistics appear here after your first
          recognition.
        </p>
      </Panel>
    )
  }

  const cards = [
    { label: 'Total runs', value: stats.total_runs },
    { label: 'Image runs', value: stats.image_runs },
    { label: 'Video runs', value: stats.video_runs },
    { label: 'Plates detected', value: stats.plates_detected },
    {
      label: 'Text rate',
      value:
        stats.ocr_text_rate == null
          ? '—'
          : `${(stats.ocr_text_rate * 100).toFixed(1)}%`,
    },
    {
      label: 'Avg confidence',
      value:
        stats.avg_best_confidence == null
          ? '—'
          : `${(stats.avg_best_confidence * 100).toFixed(1)}%`,
    },
    {
      label: 'Mean latency',
      value:
        stats.latency?.mean_seconds == null
          ? '—'
          : `${stats.latency.mean_seconds.toFixed(2)}s`,
    },
    {
      label: 'p95 latency',
      value:
        stats.latency?.p95_seconds == null
          ? '—'
          : `${stats.latency.p95_seconds.toFixed(2)}s`,
    },
  ]

  return (
    <Panel title="System statistics" subtitle="From your recorded recognition runs">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {cards.map((c) => (
          <div
            key={c.label}
            className="rounded-md border border-hairline-soft bg-panel-raised/40 p-3"
          >
            <div className="font-mono text-lg text-ink">{c.value}</div>
            <div className="mt-1 text-[10px] uppercase tracking-wider text-ink-faint">
              {c.label}
            </div>
          </div>
        ))}
      </div>

      {Object.keys(stats.status_distribution || {}).length > 0 && (
        <div className="mt-4 flex flex-wrap gap-2">
          {Object.entries(stats.status_distribution).map(([status, count]) => (
            <span
              key={status}
              className="flex items-center gap-1.5 rounded-md border border-hairline-soft px-2.5 py-1 text-[11px] text-ink-dim"
            >
              <StatusBadge status={status} />
              <span className="font-mono">×{count}</span>
            </span>
          ))}
        </div>
      )}
    </Panel>
  )
}

export default function Detection() {
  const [history, setHistory] = useState(null)
  const [filter, setFilter] = useState('ALL')
  const [searchInput, setSearchInput] = useState('')
  const [search, setSearch] = useState('')
  const [stats, setStats] = useState(null)

  // Debounce the search box so typing does not fire a request per key.
  useEffect(() => {
    const t = setTimeout(() => setSearch(searchInput.trim()), 300)
    return () => clearTimeout(t)
  }, [searchInput])

  useEffect(() => {
    const params = {}
    if (filter !== 'ALL') params.type = filter
    if (search) params.search = search
    api
      .history(params)
      .then(setHistory)
      .catch(() => setHistory({ total: 0, results: [] }))
  }, [filter, search])

  useEffect(() => {
    api
      .historyStats()
      .then(setStats)
      .catch(() => setStats(null))
  }, [])

  function exportParams(format) {
    const params = { format }
    if (filter !== 'ALL') params.type = filter
    if (search) params.search = search
    return params
  }

  return (
    <div>
      <Header
        title="Detection"
        description="Every run submitted through Live / Image Analysis or Video Analytics — recorded as it happens, nothing backfilled."
      />

      <div className="space-y-4 px-8 py-6">
        <LiveStats stats={stats} />

        <div className="flex flex-wrap items-center gap-2">
          {FILTERS.map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`rounded-md px-2.5 py-1.5 text-[12px] font-medium capitalize transition-colors ${
                filter === f ? 'bg-panel-raised text-ink' : 'text-ink-dim hover:text-ink'
              }`}
            >
              {f === 'ALL' ? 'All runs' : `${f} runs`}
            </button>
          ))}

          <div className="relative ml-2">
            <Search
              size={13}
              className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-ink-faint"
            />
            <input
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              placeholder="Search filename or plate text…"
              className="w-56 rounded-md border border-hairline bg-panel py-1.5 pl-8 pr-3 text-[12px] text-ink placeholder:text-ink-faint focus:border-signal/50 focus:outline-none"
            />
          </div>

          <div className="ml-auto flex items-center gap-2">
            {history && (
              <span className="font-mono text-[11px] text-ink-faint">
                {history.total} recorded
              </span>
            )}

            <a
              href={api.historyExportUrl(exportParams('csv'))}
              download
              title="Download the filtered history as CSV"
              className="flex items-center gap-1.5 rounded-md border border-hairline px-2.5 py-1.5 text-[11px] text-ink-dim transition-colors hover:bg-panel-raised"
            >
              <Download size={12} />
              CSV
            </a>

            <a
              href={api.historyExportUrl(exportParams('json'))}
              download
              title="Download the filtered history as JSON"
              className="flex items-center gap-1.5 rounded-md border border-hairline px-2.5 py-1.5 text-[11px] text-ink-dim transition-colors hover:bg-panel-raised"
            >
              <Download size={12} />
              JSON
            </a>
          </div>
        </div>

        {history && history.results.length === 0 && (
          <ComingNext
            icon={ScanLine}
            note={
              search
                ? `No runs match “${search}”. Try a different filename or plate text.`
                : 'No runs recorded yet. Every detection you submit on Live / Image Analysis, and every video processed here, gets logged the moment it completes — nothing is backfilled or simulated.'
            }
          />
        )}

        {history && history.results.length > 0 && (
          <Panel bodyClassName="p-0">
            <table className="w-full text-left text-[12px]">
              <thead>
                <tr className="text-ink-faint">
                  <th className="px-5 py-2 font-medium">Timestamp</th>
                  <th className="px-5 py-2 font-medium">Source</th>
                  <th className="px-5 py-2 font-medium">Detail</th>
                  <th className="px-5 py-2 font-medium">Confidence</th>
                  <th className="px-5 py-2 font-medium">Status</th>
                  <th className="px-5 py-2 font-medium">Processing time</th>
                </tr>
              </thead>
              <tbody>
                {history.results.map((r) => (
                  <tr key={r.id} className="border-t border-hairline-soft">
                    <td className="px-5 py-2 font-mono text-ink-dim">
                      {fmtTimestamp(r.timestamp)}
                    </td>
                    <td className="px-5 py-2">
                      <span className="flex items-center gap-1.5 text-ink">
                        {r.type === 'image' ? (
                          <ImageIcon size={13} className="text-ink-faint" />
                        ) : (
                          <Video size={13} className="text-ink-faint" />
                        )}
                        {r.filename || 'unknown'}
                      </span>
                    </td>
                    <td className="px-5 py-2 font-mono text-ink-dim">
                      {r.type === 'image'
                        ? r.best_plate_text
                          ? `${r.best_plate_text} · ${r.plates_detected} plate(s)`
                          : `${r.plates_detected} plate(s)`
                        : `${r.detections_found ?? 0} detections · ${
                            r.processed_frames ?? 0
                          } frames`}
                    </td>
                    <td className="px-5 py-2 font-mono text-ink-dim">
                      {r.type === 'image' && r.best_confidence !== undefined
                        ? `${(r.best_confidence * 100).toFixed(1)}%`
                        : 'N/A'}
                    </td>
                    <td className="px-5 py-2">
                      <StatusBadge status={r.status} />
                    </td>
                    <td className="px-5 py-2 font-mono text-ink-dim">
                      {r.processing_time_seconds !== undefined
                        ? `${r.processing_time_seconds}s`
                        : 'N/A'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Panel>
        )}
      </div>
    </div>
  )
}
