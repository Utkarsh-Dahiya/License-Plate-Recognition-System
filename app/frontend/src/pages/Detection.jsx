import { useEffect, useState } from 'react'
import { ScanLine, Image as ImageIcon, Video } from 'lucide-react'
import Header from '../components/Header.jsx'
import { Panel } from '../components/Panel.jsx'
import ComingNext from '../components/ComingNext.jsx'
import StatusBadge from '../components/StatusBadge.jsx'
import { api } from '../lib/api'

const FILTERS = ['ALL', 'image', 'video']

function fmtTimestamp(ts) {
  return new Date(ts * 1000).toLocaleString()
}

export default function Detection() {
  const [history, setHistory] = useState(null)
  const [filter, setFilter] = useState('ALL')

  useEffect(() => {
    const params = filter === 'ALL' ? {} : { type: filter }
    api
      .history(params)
      .then(setHistory)
      .catch(() => setHistory({ total: 0, results: [] }))
  }, [filter])

  return (
    <div>
      <Header
        title="Detection"
        description="Every run submitted through Live / Image Analysis or Video Analytics — recorded as it happens, nothing backfilled."
      />

      <div className="space-y-4 px-8 py-6">
        <div className="flex items-center gap-1">
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
          {history && (
            <span className="ml-auto font-mono text-[11px] text-ink-faint">
              {history.total} recorded
            </span>
          )}
        </div>

        {history && history.results.length === 0 && (
          <ComingNext
            icon={ScanLine}
            note="No runs recorded yet. Every detection you submit on Live / Image Analysis, and every video processed here, gets logged the moment it completes — nothing is backfilled or simulated."
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
