import { useEffect, useMemo, useState } from 'react'
import {
  Search,
  X,
  ScanLine,
  ShieldCheck,
  Activity,
  Clock3,
  Database,
  ChevronRight,
  CircleAlert,
  CheckCircle2,
  Gauge,
  Eye,
} from 'lucide-react'
import Header from '../components/Header.jsx'
import { Panel } from '../components/Panel.jsx'
import StatusBadge from '../components/StatusBadge.jsx'
import { api } from '../lib/api'

function fmtTime(s) {
  if (s === null || s === undefined || !Number.isFinite(Number(s))) return '—'

  const m = Math.floor(Number(s) / 60)
  const sec = (Number(s) % 60).toFixed(1)

  return `${m}:${sec.padStart(4, '0')}`
}

function fmtPercent(value, digits = 1) {
  const n = Number(value)

  if (!Number.isFinite(n)) return '—'

  return `${(n * 100).toFixed(digits)}%`
}

function confidenceLevel(value) {
  const n = Number(value)

  if (!Number.isFinite(n)) {
    return {
      label: 'UNKNOWN',
      text: 'text-ink-faint',
      bg: 'bg-white/[0.04]',
      bar: 'bg-ink-faint',
    }
  }

  if (n >= 0.8) {
    return {
      label: 'HIGH',
      text: 'text-signal',
      bg: 'bg-signal/[0.07]',
      bar: 'bg-signal',
    }
  }

  if (n >= 0.5) {
    return {
      label: 'REVIEW',
      text: 'text-yellow-400',
      bg: 'bg-yellow-400/[0.06]',
      bar: 'bg-yellow-400',
    }
  }

  return {
    label: 'LOW',
    text: 'text-alert',
    bg: 'bg-alert/[0.06]',
    bar: 'bg-alert',
  }
}

function ConfidenceBar({ value }) {
  const level = confidenceLevel(value)
  const numeric = Number(value)

  return (
    <div className="w-full max-w-[145px]">
      <div className="mb-1.5 flex items-center justify-between">
        <span className={`font-mono text-[11px] ${level.text}`}>
          {fmtPercent(value)}
        </span>

        <span className={`text-[9px] font-semibold tracking-[0.12em] ${level.text}`}>
          {level.label}
        </span>
      </div>

      <div className="h-1.5 overflow-hidden rounded-full bg-white/[0.06]">
        <div
          className={`h-full rounded-full ${level.bar} transition-all`}
          style={{
            width: `${Math.max(
              0,
              Math.min(100, Number.isFinite(numeric) ? numeric * 100 : 0)
            )}%`,
          }}
        />
      </div>
    </div>
  )
}

function SummaryCard({ icon: Icon, label, value, detail, accent = false }) {
  return (
    <div
      className={`rounded-xl border p-4 ${
        accent
          ? 'border-signal/20 bg-signal/[0.035]'
          : 'border-hairline-soft bg-panel-raised/30'
      }`}
    >
      <div className="flex items-start justify-between">
        <div
          className={`flex h-9 w-9 items-center justify-center rounded-lg ${
            accent
              ? 'bg-signal/10 text-signal'
              : 'bg-white/[0.035] text-ink-faint'
          }`}
        >
          <Icon size={16} strokeWidth={1.6} />
        </div>

        {accent && (
          <span className="mt-1 h-1.5 w-1.5 rounded-full bg-signal shadow-[0_0_10px_rgba(62,217,139,0.8)]" />
        )}
      </div>

      <p className="mt-4 text-[9px] font-semibold uppercase tracking-[0.14em] text-ink-faint">
        {label}
      </p>

      <p className="mt-1 font-mono text-xl font-semibold text-ink">
        {value}
      </p>

      {detail && (
        <p className="mt-1 text-[10px] text-ink-faint">
          {detail}
        </p>
      )}
    </div>
  )
}

export default function PlateRegistry() {
  const [plates, setPlates] = useState({ total: 0, results: [] })
  const [search, setSearch] = useState('')
  const [sort, setSort] = useState('confidence')
  const [statusFilter, setStatusFilter] = useState('ALL')
  const [selected, setSelected] = useState(null)
  const [detail, setDetail] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)

    const params = { sort }

    if (search.trim()) {
      params.search = search.trim()
    }

    api
      .plates(params)
      .then(setPlates)
      .catch(() => setPlates({ total: 0, results: [] }))
      .finally(() => setLoading(false))
  }, [search, sort])

  const visiblePlates = useMemo(() => {
    if (statusFilter === 'ALL') {
      return plates.results
    }

    return plates.results.filter((p) => p.status === statusFilter)
  }, [plates.results, statusFilter])

  useEffect(() => {
    if (!selected) {
      setDetail(null)
      return
    }

    setDetail(null)

    api
      .plateDetail(selected)
      .then(setDetail)
      .catch(() => setDetail(null))
  }, [selected])

  const counts = useMemo(() => {
    const results = plates.results || []

    return {
      all: results.length,
      high: results.filter((p) => p.status === 'HIGH_CONFIDENCE').length,
      review: results.filter((p) => p.status === 'REVIEW').length,
      low: results.filter((p) => p.status === 'LOW_CONFIDENCE').length,
    }
  }, [plates.results])

  return (
    <div>
      <Header
        title="Plate Registry"
        description="A searchable intelligence layer built from unique license-plate observations across the processed video run."
      />

      <div className="space-y-6 px-8 py-6">
        {/* Registry hero */}
        <div className="relative overflow-hidden rounded-2xl border border-hairline-soft bg-panel-raised/30">
          <div className="pointer-events-none absolute -right-24 -top-28 h-72 w-72 rounded-full bg-signal/[0.055] blur-3xl" />

          <div className="relative p-6 md:p-7">
            <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
              <div className="max-w-2xl">
                <div className="mb-4 flex items-center gap-2">
                  <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-signal/15 bg-signal/[0.05] text-signal">
                    <ScanLine size={17} />
                  </div>

                  <span className="font-mono text-[10px] font-semibold uppercase tracking-[0.18em] text-signal">
                    Vehicle intelligence registry
                  </span>
                </div>

                <h2 className="text-2xl font-semibold tracking-tight text-ink md:text-3xl">
                  Every observed plate,
                  <span className="text-signal"> searchable.</span>
                </h2>

                <p className="mt-3 max-w-xl text-[12px] leading-6 text-ink-dim">
                  Browse unique plate identities, inspect confidence evidence,
                  and trace individual observations back to their video frames.
                </p>
              </div>

              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:min-w-[430px]">
                <SummaryCard
                  icon={Database}
                  label="Registry"
                  value={plates.total}
                  detail="unique plates"
                  accent
                />

                <SummaryCard
                  icon={ShieldCheck}
                  label="High confidence"
                  value={counts.high}
                  detail="observations"
                />

                <SummaryCard
                  icon={Eye}
                  label="Review"
                  value={counts.review}
                  detail="need review"
                />

                <SummaryCard
                  icon={CircleAlert}
                  label="Low confidence"
                  value={counts.low}
                  detail="low evidence"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="rounded-xl border border-hairline-soft bg-panel-raised/20 p-4">
          <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
            <div className="flex flex-1 flex-col gap-3 md:flex-row">
              <div className="relative max-w-md flex-1">
                <Search
                  size={14}
                  className="absolute left-3 top-1/2 -translate-y-1/2 text-ink-faint"
                />

                <input
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search plate number..."
                  className="w-full rounded-lg border border-hairline bg-panel-raised py-2.5 pl-9 pr-9 font-mono text-[11px] text-ink outline-none placeholder:text-ink-faint transition-colors focus:border-signal/30"
                />

                {search && (
                  <button
                    onClick={() => setSearch('')}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-ink-faint hover:text-ink"
                  >
                    <X size={13} />
                  </button>
                )}
              </div>

              <select
                value={sort}
                onChange={(e) => setSort(e.target.value)}
                className="rounded-lg border border-hairline bg-panel-raised px-3 py-2.5 text-[11px] text-ink outline-none focus:border-signal/30"
              >
                <option value="confidence">Sort: Confidence</option>
                <option value="detections">Sort: Detection count</option>
                <option value="latest">Sort: Latest detection</option>
              </select>
            </div>

            <div className="flex items-center gap-2 text-[10px] text-ink-faint">
              <Activity size={13} className={loading ? 'animate-pulse text-signal' : ''} />

              <span>
                Showing{' '}
                <span className="font-mono text-ink-dim">
                  {visiblePlates.length}
                </span>{' '}
                of{' '}
                <span className="font-mono text-ink-dim">
                  {plates.total}
                </span>
              </span>
            </div>
          </div>

          {/* Status filters */}
          <div className="mt-4 flex flex-wrap gap-1.5 border-t border-hairline-soft pt-3">
            {[
              ['ALL', 'All plates', counts.all],
              ['HIGH_CONFIDENCE', 'High confidence', counts.high],
              ['REVIEW', 'Review', counts.review],
              ['LOW_CONFIDENCE', 'Low confidence', counts.low],
            ].map(([value, label, count]) => (
              <button
                key={value}
                onClick={() => setStatusFilter(value)}
                className={`rounded-lg border px-3 py-1.5 text-[10px] font-medium transition-all ${
                  statusFilter === value
                    ? 'border-signal/20 bg-signal/[0.06] text-signal'
                    : 'border-transparent text-ink-dim hover:border-hairline-soft hover:bg-white/[0.02] hover:text-ink'
                }`}
              >
                {label}
                <span className="ml-1.5 font-mono opacity-60">
                  {count}
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Registry table */}
        <Panel
          title="Unique plate identities"
          subtitle="Click any identity to inspect its complete observation history."
          bodyClassName="p-0"
        >
          <div className="overflow-x-auto">
            <table className="w-full min-w-[850px] text-left text-[11px]">
              <thead>
                <tr className="bg-white/[0.012] text-[9px] font-semibold uppercase tracking-[0.12em] text-ink-faint">
                  <th className="px-5 py-3">Plate identity</th>
                  <th className="px-5 py-3">Confidence</th>
                  <th className="px-5 py-3">Observations</th>
                  <th className="px-5 py-3">First seen</th>
                  <th className="px-5 py-3">Last seen</th>
                  <th className="px-5 py-3">Status</th>
                  <th className="px-5 py-3"></th>
                </tr>
              </thead>

              <tbody>
                {visiblePlates.slice(0, 100).map((p) => {
                  const level = confidenceLevel(p.confidence)

                  return (
                    <tr
                      key={p.plate}
                      onClick={() => setSelected(p.plate)}
                      className="group cursor-pointer border-t border-hairline-soft transition-colors hover:bg-white/[0.018]"
                    >
                      <td className="px-5 py-3.5">
                        <div className="flex items-center gap-3">
                          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-signal/10 bg-signal/[0.035] text-signal">
                            <ScanLine size={14} />
                          </div>

                          <div>
                            <p className="font-mono font-semibold tracking-wide text-ink">
                              {p.plate}
                            </p>

                            <p className="mt-0.5 text-[9px] text-ink-faint">
                              Plate identity
                            </p>
                          </div>
                        </div>
                      </td>

                      <td className="px-5 py-3.5">
                        <ConfidenceBar value={p.confidence} />
                      </td>

                      <td className="px-5 py-3.5">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-ink">
                            {p.detection_count}
                          </span>

                          <span className="text-[9px] text-ink-faint">
                            frames
                          </span>
                        </div>
                      </td>

                      <td className="px-5 py-3.5">
                        <div className="flex items-center gap-1.5 font-mono text-ink-dim">
                          <Clock3 size={11} className="text-ink-faint" />
                          {fmtTime(p.first_seen)}
                        </div>
                      </td>

                      <td className="px-5 py-3.5">
                        <div className="flex items-center gap-1.5 font-mono text-ink-dim">
                          <Clock3 size={11} className="text-ink-faint" />
                          {fmtTime(p.last_seen)}
                        </div>
                      </td>

                      <td className="px-5 py-3.5">
                        <StatusBadge status={p.status} />
                      </td>

                      <td className="px-5 py-3.5">
                        <ChevronRight
                          size={14}
                          className="text-ink-faint opacity-40 transition-all group-hover:translate-x-0.5 group-hover:text-signal group-hover:opacity-100"
                        />
                      </td>
                    </tr>
                  )
                })}

                {visiblePlates.length === 0 && (
                  <tr>
                    <td colSpan="7" className="px-5 py-16 text-center">
                      <div className="mx-auto flex h-11 w-11 items-center justify-center rounded-xl border border-hairline-soft text-ink-faint">
                        <Search size={18} />
                      </div>

                      <p className="mt-4 text-[12px] font-medium text-ink">
                        No plate identities found
                      </p>

                      <p className="mt-1 text-[10px] text-ink-faint">
                        Try a different search term or status filter.
                      </p>

                      {(search || statusFilter !== 'ALL') && (
                        <button
                          onClick={() => {
                            setSearch('')
                            setStatusFilter('ALL')
                          }}
                          className="mt-4 rounded-lg border border-hairline-soft px-3 py-1.5 text-[10px] text-ink-dim hover:border-signal/20 hover:text-signal"
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

          {visiblePlates.length > 100 && (
            <div className="flex items-center justify-between border-t border-hairline-soft px-5 py-3">
              <p className="text-[10px] text-ink-faint">
                Showing the top 100 results. Use search to narrow the registry.
              </p>

              <span className="font-mono text-[10px] text-ink-dim">
                100 / {visiblePlates.length}
              </span>
            </div>
          )}
        </Panel>
      </div>

      {/* Detail drawer */}
      {selected && (
        <div
          className="fixed inset-0 z-30 flex justify-end bg-black/55 backdrop-blur-[2px]"
          onClick={() => setSelected(null)}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="h-full w-full max-w-[480px] overflow-y-auto border-l border-hairline bg-panel shadow-2xl"
          >
            {/* Drawer header */}
            <div className="sticky top-0 z-10 border-b border-hairline-soft bg-panel/95 px-6 py-5 backdrop-blur-xl">
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-signal/15 bg-signal/[0.05] text-signal">
                    <ScanLine size={18} />
                  </div>

                  <div>
                    <p className="font-mono text-xl font-semibold tracking-wide text-ink">
                      {selected}
                    </p>

                    <p className="mt-0.5 text-[9px] uppercase tracking-[0.14em] text-ink-faint">
                      Plate intelligence record
                    </p>
                  </div>
                </div>

                <button
                  onClick={() => setSelected(null)}
                  className="flex h-8 w-8 items-center justify-center rounded-lg text-ink-faint transition-colors hover:bg-white/[0.04] hover:text-ink"
                >
                  <X size={16} />
                </button>
              </div>
            </div>

            {detail ? (
              <div className="space-y-6 p-6">
                {/* Confidence overview */}
                <div className="rounded-xl border border-signal/15 bg-signal/[0.025] p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-[9px] font-semibold uppercase tracking-[0.14em] text-ink-faint">
                        Recognition confidence
                      </p>

                      <p className="mt-1 font-mono text-2xl font-semibold text-signal">
                        {fmtPercent(detail.plate.final_confidence)}
                      </p>
                    </div>

                    <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-signal/10 text-signal">
                      <ShieldCheck size={19} />
                    </div>
                  </div>

                  <div className="mt-4 h-2 overflow-hidden rounded-full bg-white/[0.06]">
                    <div
                      className="h-full rounded-full bg-signal"
                      style={{
                        width: `${Math.max(
                          0,
                          Math.min(
                            100,
                            Number(detail.plate.final_confidence) * 100 || 0
                          )
                        )}%`,
                      }}
                    />
                  </div>
                </div>

                {/* Core metrics */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="rounded-xl border border-hairline-soft bg-panel-raised/30 p-4">
                    <div className="flex items-center gap-2 text-ink-faint">
                      <Database size={13} />
                      <span className="text-[9px] uppercase tracking-[0.12em]">
                        Detections
                      </span>
                    </div>

                    <p className="mt-2 font-mono text-lg text-ink">
                      {detail.plate.detections}
                    </p>
                  </div>

                  <div className="rounded-xl border border-hairline-soft bg-panel-raised/30 p-4">
                    <div className="flex items-center gap-2 text-ink-faint">
                      <Activity size={13} />
                      <span className="text-[9px] uppercase tracking-[0.12em]">
                        Status
                      </span>
                    </div>

                    <div className="mt-2">
                      <StatusBadge status={detail.plate.status} />
                    </div>
                  </div>

                  <div className="rounded-xl border border-hairline-soft bg-panel-raised/30 p-4">
                    <div className="flex items-center gap-2 text-ink-faint">
                      <Clock3 size={13} />
                      <span className="text-[9px] uppercase tracking-[0.12em]">
                        First seen
                      </span>
                    </div>

                    <p className="mt-2 font-mono text-sm text-ink">
                      {fmtTime(detail.plate.first_timestamp)}
                    </p>
                  </div>

                  <div className="rounded-xl border border-hairline-soft bg-panel-raised/30 p-4">
                    <div className="flex items-center gap-2 text-ink-faint">
                      <Gauge size={13} />
                      <span className="text-[9px] uppercase tracking-[0.12em]">
                        Last seen
                      </span>
                    </div>

                    <p className="mt-2 font-mono text-sm text-ink">
                      {fmtTime(detail.plate.last_timestamp)}
                    </p>
                  </div>
                </div>

                {/* Observation history */}
                <div>
                  <div className="mb-3 flex items-end justify-between">
                    <div>
                      <p className="text-[12px] font-medium text-ink">
                        Observation history
                      </p>

                      <p className="mt-1 text-[10px] text-ink-faint">
                        Frame-level evidence associated with this plate.
                      </p>
                    </div>

                    <span className="font-mono text-[10px] text-ink-faint">
                      {detail.detections.length} records
                    </span>
                  </div>

                  <div className="space-y-2">
                    {detail.detections.slice(0, 20).map((d, i) => (
                      <div
                        key={i}
                        className="rounded-lg border border-hairline-soft bg-panel-raised/25 p-3"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <span className="flex h-6 w-6 items-center justify-center rounded-md bg-white/[0.035] font-mono text-[9px] text-ink-faint">
                              {i + 1}
                            </span>

                            <span className="font-mono text-[10px] text-ink-dim">
                              frame {d.frame}
                            </span>
                          </div>

                          <span className="flex items-center gap-1.5 font-mono text-[10px] text-ink-faint">
                            <Clock3 size={10} />
                            {d.timestamp}s
                          </span>
                        </div>

                        <div className="mt-3 flex items-center justify-between">
                          <div className="flex items-center gap-2 text-[9px] text-ink-faint">
                            <CheckCircle2 size={11} className="text-signal" />
                            Recognition recorded
                          </div>

                          <span className="font-mono text-[11px] text-signal">
                            {fmtPercent(d.final_confidence, 0)}
                          </span>
                        </div>

                        <div className="mt-2 h-1 overflow-hidden rounded-full bg-white/[0.05]">
                          <div
                            className="h-full rounded-full bg-signal"
                            style={{
                              width: `${Math.max(
                                0,
                                Math.min(
                                  100,
                                  Number(d.final_confidence) * 100 || 0
                                )
                              )}%`,
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>

                  {detail.detections.length > 20 && (
                    <p className="mt-3 text-center text-[10px] text-ink-faint">
                      Showing first 20 of {detail.detections.length} observations.
                    </p>
                  )}
                </div>
              </div>
            ) : (
              <div className="flex min-h-[300px] flex-col items-center justify-center px-6 text-center">
                <LoaderIcon />

                <p className="mt-4 text-[12px] font-medium text-ink">
                  Loading plate intelligence
                </p>

                <p className="mt-1 text-[10px] text-ink-faint">
                  Retrieving observation history…
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function LoaderIcon() {
  return (
    <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-hairline-soft bg-panel-raised">
      <Activity size={17} className="animate-pulse text-signal" />
    </div>
  )
}