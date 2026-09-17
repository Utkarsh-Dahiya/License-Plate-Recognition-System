import { useEffect, useState } from 'react'
import Header from '../components/Header.jsx'
import { Panel } from '../components/Panel.jsx'
import { api } from '../lib/api'

function Row({ label, value, ok }) {
  return (
    <div className="flex items-center justify-between border-b border-hairline-soft py-2.5 text-[12px] last:border-0">
      <span className="text-ink-dim">{label}</span>
      <span className={`font-mono ${ok === false ? 'text-alert' : 'text-ink'}`}>{value}</span>
    </div>
  )
}

export default function System() {
  const [system, setSystem] = useState(null)
  const [health, setHealth] = useState(null)

  useEffect(() => {
    api.system().then(setSystem).catch(() => setSystem(null))
    api.health().then(setHealth).catch(() => setHealth(null))
  }, [])

  return (
    <div>
      <Header title="System" description="Engine status, data sources, and stack information." />

      <div className="grid grid-cols-1 gap-4 px-8 py-6 lg:grid-cols-2">
        <Panel title="Stack">
          {system &&
            Object.entries(system.stack).map(([k, v]) => (
              <Row key={k} label={k.replaceAll('_', ' ')} value={v} />
            ))}
        </Panel>

        <Panel title="Engines">
          <Row label="Backend" value={health?.status === 'online' ? 'Online' : 'Offline'} ok={health?.status === 'online'} />
          <Row
            label="YOLO / EasyOCR"
            value={
              health?.models_loaded
                ? 'Ready'
                : health?.status === 'online'
                  ? 'Idle — load on first image'
                  : 'Offline'
            }
            ok={health?.models_loaded ? true : health?.status === 'online' ? undefined : false}
          />
        </Panel>

        <Panel title="Data sources" className="lg:col-span-2" bodyClassName="p-0">
          <table className="w-full text-left text-[12px]">
            <thead>
              <tr className="text-ink-faint">
                <th className="px-5 py-2 font-medium">File</th>
                <th className="px-5 py-2 font-medium">Path</th>
                <th className="px-5 py-2 font-medium">Status</th>
                <th className="px-5 py-2 font-medium">Modified</th>
              </tr>
            </thead>
            <tbody>
              {system &&
                Object.entries(system.sources).map(([key, info]) => (
                  <tr key={key} className="border-t border-hairline-soft">
                    <td className="px-5 py-2 text-ink">{key.replaceAll('_', ' ')}</td>
                    <td className="max-w-xs truncate px-5 py-2 font-mono text-[11px] text-ink-faint">
                      {info.path}
                    </td>
                    <td className="px-5 py-2">
                      <span className={info.exists ? 'text-signal' : 'text-alert'}>
                        {info.exists ? 'Found' : 'Missing'}
                      </span>
                    </td>
                    <td className="px-5 py-2 font-mono text-ink-dim">{info.modified || '—'}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </Panel>
      </div>
    </div>
  )
}
