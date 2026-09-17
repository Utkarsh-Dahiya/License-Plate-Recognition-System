export default function Header({ title, description, actions }) {
  return (
    <header className="flex items-start justify-between border-b border-hairline px-8 py-6">
      <div>
        <h1 className="font-display text-xl font-bold tracking-tight text-ink">{title}</h1>
        {description && (
          <p className="mt-1 max-w-xl text-[13px] leading-relaxed text-ink-dim">
            {description}
          </p>
        )}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </header>
  )
}
