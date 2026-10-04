import { NavLink } from 'react-router-dom'
import {
  LayoutGrid,
  ScanLine,
  ImagePlay,
  Layers,
  Video,
  BookMarked,
  Gauge,
  Settings2,
} from 'lucide-react'
import { useEffect, useState } from 'react'
import { api } from '../lib/api'

const NAV_ITEMS = [
  { to: '/', label: 'Overview', icon: LayoutGrid, end: true },
  { to: '/detection', label: 'Detection', icon: ScanLine },
  { to: '/image-analysis', label: 'Live / Image Analysis', icon: ImagePlay },
  { to: '/batch', label: 'Batch Intelligence', icon: Layers },
  { to: '/video', label: 'Video Analytics', icon: Video },
  { to: '/registry', label: 'Plate Registry', icon: BookMarked },
  { to: '/model', label: 'Model Performance', icon: Gauge },
  { to: '/system', label: 'System', icon: Settings2 },
]

function EngineDot({ ok, idle }) {
  return (
    <span
      className={`inline-block h-1.5 w-1.5 rounded-full ${
        ok ? 'bg-signal' : idle ? 'bg-review' : 'bg-ink-faint'
      }`}
    />
  )
}

export default function Sidebar() {
  const [health, setHealth] = useState(null)

  useEffect(() => {
    let cancelled = false
    api
      .health()
      .then((d) => !cancelled && setHealth(d))
      .catch(() => !cancelled && setHealth({ status: 'offline' }))
    return () => {
      cancelled = true
    }
  }, [])

  const online = health?.status === 'online'
  const modelsReady = !!health?.models_loaded
  const modelsIdle = online && !modelsReady

  return (
    <aside className="flex h-full w-14 shrink-0 flex-col overflow-y-auto border-r border-hairline bg-[#0D1219] md:w-60">
      <div className="flex items-center gap-2.5 px-3 py-5 md:px-5 md:py-6">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-hairline bg-panel font-mono text-sm font-semibold text-signal">
          LPD
        </div>
        {/* Labels collapse with the rail on narrow viewports. */}
        <div className="hidden min-w-0 leading-tight md:block">
          <div className="truncate font-display text-[13px] font-bold tracking-tight text-ink">
            License Plate Detection
          </div>
          <div className="font-mono text-[10px] text-ink-faint">OCR System</div>
        </div>
      </div>

      <nav aria-label="Main navigation" className="flex-1 space-y-0.5 px-2 md:px-3">
        {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            title={label}
            className={({ isActive }) =>
              `flex items-center gap-2.5 rounded-md px-2.5 py-2 text-[13px] transition-colors md:px-3 ${
                isActive
                  ? 'bg-panel-raised text-ink'
                  : 'text-ink-dim hover:bg-panel-raised/60 hover:text-ink'
              }`
            }
          >
            {({ isActive }) => (
              <>
                <Icon
                  size={16}
                  strokeWidth={2}
                  aria-hidden="true"
                  className={`shrink-0 ${isActive ? 'text-signal' : 'text-ink-faint'}`}
                />
                <span className="hidden truncate md:inline">{label}</span>
              </>
            )}
          </NavLink>
        ))}
      </nav>

      <div className="hidden space-y-2 border-t border-hairline px-5 py-4 md:block">
        <div className="flex items-center justify-between">
          <span className="text-[11px] text-ink-dim">System</span>
          <span className="flex items-center gap-1.5 font-mono text-[11px] text-ink-dim">
            <EngineDot ok={online} />
            {online ? 'Online' : 'Offline'}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-[11px] text-ink-dim">YOLO Engine</span>
          <span className="flex items-center gap-1.5 font-mono text-[11px] text-ink-dim">
            <EngineDot ok={modelsReady} idle={modelsIdle} />
            {modelsReady ? 'Ready' : online ? 'Idle' : '—'}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-[11px] text-ink-dim">EasyOCR Engine</span>
          <span className="flex items-center gap-1.5 font-mono text-[11px] text-ink-dim">
            <EngineDot ok={modelsReady} idle={modelsIdle} />
            {modelsReady ? 'Ready' : online ? 'Idle' : '—'}
          </span>
        </div>
        {modelsIdle && (
          <p className="text-[10px] leading-4 text-ink-faint">
            Idle — load on first image
          </p>
        )}
      </div>

      {/* Compact status indicator for the narrow rail */}
      <div className="flex justify-center border-t border-hairline py-3 md:hidden">
        <EngineDot ok={online} idle={modelsIdle} />
        <span className="sr-only">
          Backend {online ? 'online' : 'offline'}, models{' '}
          {modelsReady ? 'ready' : 'not loaded'}
        </span>
      </div>
    </aside>
  )
}
