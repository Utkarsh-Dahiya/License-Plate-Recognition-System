import { ShieldCheck, AlertTriangle, CircleAlert, Sparkles } from 'lucide-react'

export default function IndianPlateCard({ plate, isSelected = false, className = '' }) {
  if (!plate) return null

  const text = (plate.ocr_text || '').trim()
  const hasText = text.length > 0
  const stateCode = plate.state_code || (hasText && text.length >= 2 ? text.slice(0, 2) : null)
  const stateName = plate.state_name || null
  const conf = Number(plate.final_confidence || 0)
  const isStrict = plate.strict_format === true

  return (
    <div
      className={`relative inline-flex flex-col overflow-hidden rounded-md border font-mono select-none transition-all ${
        isSelected
          ? 'border-signal ring-2 ring-signal/40 shadow-lg shadow-signal/15'
          : 'border-zinc-700 hover:border-zinc-500'
      } ${className}`}
      style={{
        background: 'linear-gradient(180deg, #181c22 0%, #0e1217 100%)',
      }}
    >
      {/* Plate Frame */}
      <div className="flex items-stretch min-h-[52px]">
        {/* Blue IND Strip */}
        <div className="flex w-7 shrink-0 flex-col items-center justify-between bg-[#003893] py-1 text-white border-r border-[#002868]">
          <div className="h-2 w-2 rounded-full border border-yellow-300/80 bg-yellow-400/20 flex items-center justify-center">
            <span className="block h-0.5 w-0.5 rounded-full bg-yellow-300" />
          </div>
          <span className="text-[9px] font-black tracking-widest text-white">IND</span>
        </div>

        {/* Plate Number & Registration Content */}
        <div className="flex flex-1 flex-col justify-center px-3.5 py-1.5 bg-[#12171e]">
          {hasText ? (
            <div className="font-mono text-xl sm:text-2xl font-extrabold tracking-[0.18em] text-[#f4f7fa] drop-shadow-[0_1px_2px_rgba(0,0,0,0.8)]">
              {text}
            </div>
          ) : (
            <div className="font-mono text-sm font-semibold tracking-wider text-alert flex items-center gap-1.5">
              <CircleAlert size={14} />
              TEXT NOT READABLE
            </div>
          )}

          {/* State & Series Subtitle */}
          {(stateName || stateCode) && (
            <div className="mt-0.5 flex items-center gap-2 text-[10px] font-mono font-medium text-ink-dim tracking-wider uppercase">
              <span className="text-signal">{stateCode}</span>
              {stateName && <span>· {stateName}</span>}
              {isStrict && (
                <span className="rounded bg-signal/10 px-1 py-0.2 text-[9px] text-signal font-semibold border border-signal/20">
                  Standard Format
                </span>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
