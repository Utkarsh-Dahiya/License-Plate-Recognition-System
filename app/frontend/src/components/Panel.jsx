export function Panel({ title, subtitle, children, className = '', bodyClassName = '' }) {
  return (
    <div className={`rounded-lg border border-hairline bg-panel ${className}`}>
      {(title || subtitle) && (
        <div className="border-b border-hairline-soft px-5 py-4">
          {title && <h3 className="text-[13px] font-semibold text-ink">{title}</h3>}
          {subtitle && <p className="mt-0.5 text-[12px] text-ink-dim">{subtitle}</p>}
        </div>
      )}
      <div className={bodyClassName || 'p-5'}>{children}</div>
    </div>
  )
}
