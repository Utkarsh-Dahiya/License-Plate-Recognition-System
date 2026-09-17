export default function ComingNext({ icon: Icon, note }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-hairline px-8 py-20 text-center">
      {Icon && <Icon size={22} className="mb-4 text-ink-faint" strokeWidth={1.5} />}
      <p className="max-w-sm text-[13px] leading-relaxed text-ink-dim">{note}</p>
    </div>
  )
}
