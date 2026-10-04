export default function Header({ title, description, actions }) {
  return (
    <header className="flex flex-wrap items-start justify-between gap-3 border-b border-hairline px-4 py-5 sm:px-6 lg:px-8 lg:py-6">
      <div className="min-w-0">
        <h1 className="font-display text-xl font-bold tracking-tight text-ink">{title}</h1>
        {description && (
          <p className="mt-1 max-w-xl text-[13px] leading-relaxed text-ink-dim">
            {description}
          </p>
        )}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </header>
  )
}
