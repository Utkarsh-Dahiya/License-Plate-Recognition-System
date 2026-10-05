import { useEffect, useMemo, useRef, useState } from 'react'
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
  Loader2,
} from 'lucide-react'
import Header from '../components/Header.jsx'
import { Panel } from '../components/Panel.jsx'
import StatusBadge from '../components/StatusBadge.jsx'
import ErrorNotice from '../components/ErrorNotice.jsx'
import { api } from '../lib/api'
import { useAsyncResource, useDebouncedValue } from '../hooks/useAsyncResource'

// The search box is instant, but the API only sees the value once it settles,
// so typing never fires a request per keystroke.
const SEARCH_DEBOUNCE_MS = 300

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
  const [search, setSearch] = useState('')
  const [sort, setSort] = useState('confidence')
  const [statusFilter, setStatusFilter] = useState('ALL')
  const [selected, setSelected] = useState(null)

  const settledSearch = useDebouncedValue(search, SEARCH_DEBOUNCE_MS)

  // loading / error / retry / out-of-order-reply protection for the list.
  // A failed fetch is now visible (and retryable) instead of being masked as
  // "No plate identities found".
  const registry = useAsyncResource(
    () => {
      const params = { sort }
      const needle = settledSearch.trim()
      if (needle) params.search = needle
      return api.plates(params)
    },
    [sort, settledSearch],
    { initialData: { total: 0, results: [] } },
  )

  const plates = registry.data

  // Driven by `selected`: resolves to null when the drawer closes, and a
  // failed fetch surfaces an error with retry — it used to spin forever.
  const detailResource = useAsyncResource(
    () => (selected ? api.plateDetail(selected) : Promise.resolve(null)),
    [selected],
  )

  // The drawer below keeps reading `detail` as plain data, unchanged.
  // Stale-while-revalidate keeps the PREVIOUS plate's payload in `data`, so it
  // is only trusted while `dataKey` still matches the open selection —
  // otherwise a failed fetch would render another plate's record.
  const detail = detailResource.dataKey === selected ? detailResource.data : null

  const visiblePlates = useMemo(() => {
    if (statusFilter === 'ALL') {
      return plates.results
    }

    return plates.results.filter((p) => p.status === statusFilter)
  }, [plates.results, statusFilter])

  const panelRef = useRef(null)

  // Dialog behaviour: Escape closes, and focus moves into the drawer when it
  // opens so keyboard users are not left behind it.
  useEffect(() => {
    if (!selected) return undefined

    panelRef.current?.focus()

    const onKeyDown = (e) => {
      if (e.key === 'Escape') setSelected(null)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
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
                  aria-label="Search plate number"
                  placeholder="Search plate number..."
                  className="w-full rounded-lg border border-hairline bg-panel-raised py-2.5 pl-9 pr-9 font-mono text-[11px] text-ink outline-none placeholder:text-ink-faint transition-colors focus:border-signal/30"
                />

                {search && (
                  <button
                    type="button"
                    onClick={() => setSearch('')}
                    aria-label="Clear search"
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-ink-faint hover:text-ink"
                  >
                    <X size={13} aria-hidden="true" />
                  </button>
                )}
              </div>

              <select
                value={sort}
                onChange={(e) => setSort(e.target.value)}
                aria-label="Sort plate registry"
                className="rounded-lg border border-hairline bg-panel-raised px-3 py-2.5 text-[11px] text-ink outline-none focus:border-signal/30"
              >
                <option value="confidence">Sort: Confidence</option>
                <option value="detections">Sort: Detection count</option>
                <option value="latest">Sort: Latest detection</option>
              </select>
            </div>

            <div className="flex items-center gap-2 text-[10px] text-ink-faint">
              <Activity
                size={13}
                className={registry.loading ? 'animate-pulse text-signal' : ''}
                aria-hidden="true"
              />

              <span role="status">
                {registry.loading && plates.total === 0 ? (
                  'Loading registry…'
                ) : (
                  <>
                    Showing{' '}
                    <span className="font-mono text-ink-dim">
                      {visiblePlates.length}
                    </span>{' '}
                    of{' '}
                    <span className="font-mono text-ink-dim">
                      {plates.total}
                    </span>
                  </>
                )}
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
                type="button"
                onClick={() => setStatusFilter(value)}
                aria-pressed={statusFilter === value}
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
          {/* A failed list fetch is reported with a retry instead of being
              shown as an empty registry. */}
          {registry.error && (
            <div className="px-5 py-4">
              <ErrorNotice
                error={registry.error}
                onRetry={registry.retry}
                retryLabel="Reload registry"
              />
            </div>
          )}

          <div
            className="overflow-x-auto"
            role="region"
            aria-label="Unique plate identities"
            aria-busy={registry.loading}
            tabIndex={0}
          >
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
                        {/* A real button makes "inspect" keyboard-reachable;
                            the row onClick above still serves mouse users. */}
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation()
                            setSelected(p.plate)
                          }}
                          aria-label={`Inspect ${p.plate}`}
                          className="flex h-7 w-7 items-center justify-center rounded-lg text-ink-faint transition-all hover:bg-white/[0.04] hover:text-signal focus:outline-none focus-visible:ring-1 focus-visible:ring-signal/50"
                        >
                          <ChevronRight
                            size={14}
                            aria-hidden="true"
                            className="opacity-40 transition-all group-hover:translate-x-0.5 group-hover:text-signal group-hover:opacity-100"
                          />
                        </button>
                      </td>
                    </tr>
                  )
                })}

                {registry.loading && visiblePlates.length === 0 && (
                  <tr>
                    <td colSpan="7" className="px-5 py-16 text-center">
                      <span
                        role="status"
                        className="inline-flex items-center gap-2 text-ink-faint"
                      >
                        <Loader2 size={13} className="animate-spin" aria-hidden="true" />
                        Loading plate identities…
                      </span>
                    </td>
                  </tr>
                )}

                {!registry.loading && !registry.error && visiblePlates.length === 0 && (
                  <tr>
                    <td colSpan="7" className="px-5 py-16 text-center">
                      <div className="mx-auto flex h-11 w-11 items-center justify-center rounded-xl border border-hairline-soft text-ink-faint">
                        <Search size={18} />
                      </div>

                      <p role="status" className="mt-4 text-[12px] font-medium text-ink">
                        No plate identities found
                      </p>

                      <p className="mt-1 text-[10px] text-ink-faint">
                        Try a different search term or status filter.
                      </p>

                      {(search || statusFilter !== 'ALL') && (
                        <button
                          type="button"
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
            role="dialog"
            aria-modal="true"
            aria-label={`Plate intelligence record for ${selected}`}
            tabIndex={-1}
            ref={panelRef}
            onClick={(e) => e.stopPropagation()}
            className="h-full w-full max-w-[480px] overflow-y-auto border-l border-hairline bg-panel shadow-2xl focus:outline-none"
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
                  type="button"
                  onClick={() => setSelected(null)}
                  aria-label="Close plate details"
                  className="flex h-8 w-8 items-center justify-center rounded-lg text-ink-faint transition-colors hover:bg-white/[0.04] hover:text-ink"
                >
                  <X size={16} aria-hidden="true" />
                </button>
              </div>
            </div>

            {/* A failed detail read used to leave this drawer spinning
                forever; it now reports and offers a retry. */}
            {detailResource.error && !detail ? (
              <div className="p-6">
                <ErrorNotice
                  error={detailResource.error}
                  onRetry={detailResource.retry}
                  retryLabel="Reload record"
                />
              </div>
            ) : detail ? (
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
              <div
                role="status"
                aria-live="polite"
                className="flex min-h-[300px] flex-col items-center justify-center px-6 text-center"
              >
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