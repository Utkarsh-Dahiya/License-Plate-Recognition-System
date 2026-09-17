/**
 * A single bordered rail of stats separated by hairlines, rather than a
 * grid of individually-shadowed cards. Each stat reads like an instrument
 * readout: a large mono figure with a quiet label underneath.
 */
export function StatRail({ stats }) {
  return (
    <div className="grid grid-cols-2 divide-x divide-y divide-hairline-soft rounded-lg border border-hairline bg-panel sm:grid-cols-3 lg:grid-cols-6 lg:divide-y-0">
      {stats.map(({ label, value, unit, tone }) => (
        <div key={label} className="px-5 py-4">
          <div
            className={`font-mono text-2xl font-medium tabular-nums ${
              tone === 'signal'
                ? 'text-signal'
                : tone === 'alert'
                ? 'text-alert'
                : 'text-ink'
            }`}
          >
            {value}
            {unit && <span className="ml-0.5 text-sm text-ink-faint">{unit}</span>}
          </div>
          <div className="mt-1 text-[12px] text-ink-dim">{label}</div>
        </div>
      ))}
    </div>
  )
}
