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
    <aside className="flex h-full w-60 shrink-0 flex-col border-r border-hairline bg-[#0D1219]">
      <div className="flex items-center gap-2.5 px-5 py-6">
        <div className="flex h-8 w-8 items-center justify-center rounded-md border border-hairline bg-panel font-mono text-sm font-semibold text-signal">
          LV
        </div>
        <div className="leading-tight">
          <div className="font-display text-[13px] font-bold tracking-tight text-ink">
            License Vision AI
          </div>
          <div className="font-mono text-[10px] text-ink-faint">ALPR platform</div>
        </div>
      </div>

      <nav className="flex-1 space-y-0.5 px-3">
        {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              `flex items-center gap-2.5 rounded-md px-3 py-2 text-[13px] transition-colors ${
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
                  className={isActive ? 'text-signal' : 'text-ink-faint'}
                />
                {label}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      <div className="space-y-2 border-t border-hairline px-5 py-4">
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
    </aside>
  )
}
