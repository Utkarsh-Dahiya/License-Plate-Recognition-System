import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  ImagePlus,
  Video,
  TriangleAlert,
  Image as ImageIcon,
  ArrowUpRight,
  Activity,
  ScanLine,
  Database,
  ShieldCheck,
} from 'lucide-react'
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from 'recharts'
import Header from '../components/Header.jsx'
import { Panel } from '../components/Panel.jsx'
import { StatRail } from '../components/StatRail.jsx'
import StatusBadge, { confidenceBand } from '../components/StatusBadge.jsx'
import { api } from '../lib/api'

const STATUS_COLOR = {
  SUCCESS: '#3ED98B',
  OCR_FAILED: '#EF5B54',
  NO_PLATE: '#4E5964',
}

function pct(n) {
  if (n === null || n === undefined) return 'N/A'
  return `${Number(n).toFixed(1)}`
}

function confidenceLabel(value) {
  return confidenceBand(value)
}

export default function Overview() {
  const navigate = useNavigate()
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [recentRuns, setRecentRuns] = useState([])
  const [topPlates, setTopPlates] = useState([])

  useEffect(() => {
    let cancelled = false

    api
      .dashboard()
      .then((d) => !cancelled && setData(d))
      .catch((e) => !cancelled && setError(e.message))

    api
      .history({ page_size: 5 })
      .then((h) => !cancelled && setRecentRuns(h.results))
      .catch(() => {})

    api
      .plates({ sort: 'confidence' })
      .then((p) => !cancelled && setTopPlates(p.results.slice(0, 5)))
      .catch(() => {})

    return () => {
      cancelled = true
    }
  }, [])

  const stats = data
    ? [
        {
          label: 'Total images',
          value: data.total_images ?? 'N/A',
        },
        {
          label: 'Plates detected',
          value: data.plates_detected ?? 'N/A',
        },
        {
          label: 'OCR text extracted',
          value: pct(data.ocr_success_rate),
          unit: '%',
          tone: 'signal',
        },
        {
          label: 'Avg YOLO confidence',
          value: pct(data.mean_yolo_confidence),
          unit: '%',
        },
        {
          label: 'Avg OCR confidence',
          value: pct(data.mean_ocr_confidence),
          unit: '%',
        },
        {
          label: 'Unique OCR strings (video)',
          value: data.unique_plates_in_video ?? 'N/A',
        },
      ]
    : []

  const successfulReads =
    data?.status_distribution?.find((x) => x.status === 'SUCCESS')?.count ?? 0

  const failedReads =
    data?.status_distribution?.find((x) => x.status === 'OCR_FAILED')?.count ?? 0

  const noPlateReads =
    data?.status_distribution?.find((x) => x.status === 'NO_PLATE')?.count ?? 0

  return (
    <div className="pb-16">
      <Header
        title="Overview"
        description="Detection and recognition intelligence across your image batches and video runs. OCR rates below are text extraction, not verified plate accuracy."
      />

      <div className="space-y-6 px-8 py-6">

        {/* HERO */}
        <section className="relative overflow-hidden rounded-xl border border-hairline bg-panel">
          <div className="pointer-events-none absolute -right-20 -top-20 h-64 w-64 rounded-full bg-signal/5 blur-3xl" />
          <div className="pointer-events-none absolute -bottom-24 left-1/3 h-48 w-48 rounded-full bg-signal/5 blur-3xl" />

          <div className="relative grid grid-cols-1 lg:grid-cols-[1.45fr,0.75fr]">
            <div className="border-b border-hairline p-7 lg:border-b-0 lg:border-r lg:p-9">
              <div className="mb-4 flex items-center gap-2">
                <span className="flex h-7 w-7 items-center justify-center rounded-md bg-signal/10 text-signal">
                  <ScanLine size={15} />
                </span>

                <span className="text-[10px] font-semibold uppercase tracking-[0.2em] text-signal">
                  AI License Plate Intelligence
                </span>
              </div>

              <h2 className="max-w-2xl font-display text-3xl font-bold tracking-tight text-ink sm:text-4xl">
                Turn visual data into
                <span className="text-signal"> searchable intelligence.</span>
              </h2>

              <p className="mt-4 max-w-xl text-[13px] leading-6 text-ink-dim">
                Detect license plates with YOLO, extract text using EasyOCR,
                validate recognition confidence, and transform images and
                video into a structured plate registry.
              </p>

              <div className="mt-7 flex flex-wrap items-center gap-3">
                <button
                  onClick={() => navigate('/image-analysis')}
                  className="group flex items-center gap-2 rounded-md bg-signal px-4 py-2.5 text-[13px] font-semibold text-[#062015] transition-all hover:bg-signal/90"
                >
                  <ImagePlus size={15} strokeWidth={2.5} />
                  Analyze Image
                  <ArrowUpRight
                    size={14}
                    className="transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5"
                  />
                </button>

                <button
                  onClick={() => navigate('/video')}
                  className="flex items-center gap-2 rounded-md border border-hairline px-4 py-2.5 text-[13px] font-medium text-ink transition-colors hover:bg-panel-raised"
                >
                  <Video size={15} />
                  Analyze Video
                </button>
              </div>
            </div>

            <div className="p-7">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-ink-faint">
                    Processing pipeline
                  </div>

                  <div className="mt-1 text-[12px] text-ink-dim">
                    End-to-end recognition
                  </div>
                </div>

                <Activity size={16} className="text-signal" />
              </div>

              <div className="mt-6 space-y-3">
                {[
                  ['01', 'Input', 'Image / Video'],
                  ['02', 'Detection', 'YOLO'],
                  ['03', 'Recognition', 'EasyOCR'],
                  ['04', 'Validation', 'Format scoring'],
                  ['05', 'Registry', 'Searchable output'],
                ].map(([number, title, subtitle], index) => (
                  <div key={number} className="flex items-center gap-3">
                    <span className="w-5 font-mono text-[10px] text-ink-faint">
                      {number}
                    </span>

                    <div
                      className={`flex h-7 w-7 items-center justify-center rounded-md border ${
                        index === 4
                          ? 'border-signal-dim bg-signal/10 text-signal'
                          : 'border-hairline-soft bg-panel-raised text-ink-dim'
                      }`}
                    >
                      {index === 4 ? (
                        <Database size={13} />
                      ) : (
                        <span className="h-1.5 w-1.5 rounded-full bg-current" />
                      )}
                    </div>

                    <div>
                      <div className="text-[12px] text-ink">{title}</div>
                      <div className="text-[10px] text-ink-faint">
                        {subtitle}
                      </div>
                    </div>

                    {index < 4 && (
                      <div className="ml-auto h-px w-5 bg-hairline-soft" />
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* KPI RAIL */}
        {error ? (
          <Panel>
            <div className="flex items-center gap-2 text-[13px] text-alert">
              <TriangleAlert size={15} />
              Couldn't reach the backend ({error}). Is `uvicorn main:app`
              running on :8000?
            </div>
          </Panel>
        ) : data ? (
          <StatRail stats={stats} />
        ) : (
          <div className="h-24 animate-pulse rounded-lg border border-hairline bg-panel" />
        )}

        {/* LIVE SYSTEM SUMMARY */}
        {data && (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <div className="rounded-lg border border-hairline bg-panel p-4">
              <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-ink-faint">
                <ShieldCheck size={13} className="text-signal" />
                Successful OCR
              </div>

              <div className="mt-3 font-mono text-2xl text-ink">
                {successfulReads}
              </div>

              <div className="mt-1 text-[11px] text-ink-faint">
                Images where EasyOCR returned some text (not plate-text accuracy)
              </div>
            </div>

            <div className="rounded-lg border border-hairline bg-panel p-4">
              <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-ink-faint">
                <TriangleAlert size={13} className="text-amber-300" />
                OCR failed (no text)
              </div>

              <div className="mt-3 font-mono text-2xl text-ink">
                {failedReads}
              </div>

              <div className="mt-1 text-[11px] text-ink-faint">
                Images with a plate box but no OCR text
              </div>
            </div>

            <div className="rounded-lg border border-hairline bg-panel p-4">
              <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-ink-faint">
                <ScanLine size={13} className="text-ink-dim" />
                No plate detected
              </div>

              <div className="mt-3 font-mono text-2xl text-ink">
                {noPlateReads}
              </div>

              <div className="mt-1 text-[11px] text-ink-faint">
                Images without a detected plate
              </div>
            </div>
          </div>
        )}

        {/* CHARTS */}
        {data && (
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">

            <Panel
              title="Detection status"
              subtitle="Outcome distribution across processed images"
            >
              <div className="flex items-center gap-4">
                <ResponsiveContainer width="55%" height={210}>
                  <PieChart>
                    <Pie
                      data={data.status_distribution}
                      dataKey="count"
                      nameKey="status"
                      innerRadius={58}
                      outerRadius={82}
                      paddingAngle={3}
                      stroke="none"
                    >
                      {data.status_distribution.map((entry) => (
                        <Cell
                          key={entry.status}
                          fill={STATUS_COLOR[entry.status] || '#4E5964'}
                        />
                      ))}
                    </Pie>

                    <Tooltip
                      contentStyle={{
                        background: '#151C27',
                        border: '1px solid #232B36',
                        borderRadius: 6,
                        fontSize: 12,
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>

                <div className="min-w-0 flex-1 space-y-3">
                  {data.status_distribution.map((entry) => (
                    <div
                      key={entry.status}
                      className="flex items-center gap-2 text-[12px]"
                    >
                      <span
                        className="h-2 w-2 shrink-0 rounded-full"
                        style={{
                          backgroundColor:
                            STATUS_COLOR[entry.status] || '#4E5964',
                        }}
                      />

                      <span className="truncate text-ink-dim">
                        {entry.status.replaceAll('_', ' ')}
                      </span>

                      <span className="ml-auto font-mono text-ink">
                        {entry.count}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </Panel>

            <Panel
              title="OCR confidence distribution"
              subtitle="Observed confidence across successful reads"
            >
              <ResponsiveContainer width="100%" height={210}>
                <BarChart data={data.ocr_confidence_distribution}>
                  <CartesianGrid
                    stroke="#1A2129"
                    vertical={false}
                  />

                  <XAxis
                    dataKey="range"
                    tick={{ fill: '#8B96A3', fontSize: 11 }}
                    axisLine={{ stroke: '#232B36' }}
                    tickLine={false}
                  />

                  <YAxis
                    tick={{ fill: '#8B96A3', fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                    width={30}
                  />

                  <Tooltip
                    cursor={{ fill: '#151C27' }}
                    contentStyle={{
                      background: '#151C27',
                      border: '1px solid #232B36',
                      borderRadius: 6,
                      fontSize: 12,
                    }}
                  />

                  <Bar
                    dataKey="count"
                    fill="#5B8DEF"
                    radius={[4, 4, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </Panel>
          </div>
        )}

        {/* RECENT ACTIVITY */}
        {(recentRuns.length > 0 || topPlates.length > 0) && (
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">

            {recentRuns.length > 0 && (
              <Panel
                title="Recent activity"
                subtitle="Latest detection runs"
                bodyClassName="p-0"
              >
                <div className="divide-y divide-hairline-soft">
                  {recentRuns.map((r) => (
                    <div
                      key={r.id}
                      className="flex items-center gap-3 px-5 py-3"
                    >
                      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-hairline-soft bg-panel-raised">
                        {r.type === 'image' ? (
                          <ImageIcon size={13} className="text-ink-faint" />
                        ) : (
                          <Video size={13} className="text-ink-faint" />
                        )}
                      </div>

                      <div className="min-w-0 flex-1">
                        <div className="truncate text-[12px] text-ink">
                          {r.filename || 'unknown'}
                        </div>

                        <div className="mt-0.5 font-mono text-[10px] text-ink-faint">
                          {r.type === 'image'
                            ? r.best_plate_text || 'No OCR read'
                            : `${r.detections_found ?? 0} detections`}
                        </div>
                      </div>

                      <StatusBadge status={r.status} />
                    </div>
                  ))}
                </div>
              </Panel>
            )}

            {topPlates.length > 0 && (
              <Panel
                title="Top recognized plates"
                subtitle="Highest-confidence unique OCR strings from the saved video run"
                bodyClassName="p-0"
              >
                <div className="divide-y divide-hairline-soft">
                  {topPlates.map((p) => {
                    const confidence = Number(p.confidence ?? 0)

                    return (
                      <div
                        key={p.plate}
                        className="flex items-center gap-4 px-5 py-3"
                      >
                        <div className="flex h-9 min-w-[88px] items-center justify-center rounded-md border border-hairline-soft bg-panel-raised px-3">
                          <span className="font-mono text-[12px] font-semibold tracking-wider text-ink">
                            {p.plate}
                          </span>
                        </div>

                        <div className="min-w-0 flex-1">
                          <div className="mb-1 flex items-center justify-between">
                            <span className="text-[10px] text-ink-faint">
                              Confidence
                            </span>

                            <span className="font-mono text-[10px] text-ink-dim">
                              {(confidence * 100).toFixed(1)}%
                            </span>
                          </div>

                          <div className="h-1.5 overflow-hidden rounded-full bg-panel-raised">
                            <div
                              className="h-full rounded-full bg-signal"
                              style={{
                                width: `${Math.min(
                                  100,
                                  confidence * 100,
                                )}%`,
                              }}
                            />
                          </div>
                        </div>

                        <div className="hidden text-right sm:block">
                          <div className="font-mono text-[10px] text-ink-dim">
                            {p.detection_count}
                          </div>
                          <div className="text-[9px] uppercase tracking-wider text-ink-faint">
                            detections
                          </div>
                        </div>

                        <StatusBadge
                          status={
                            p.status ||
                            confidenceLabel(confidence)
                          }
                        />
                      </div>
                    )
                  })}
                </div>
              </Panel>
            )}
          </div>
        )}
      </div>
    </div>
  )
}