import { ScanLine, ArrowRight, ImagePlus } from 'lucide-react'

/**
 * Zero-detection state.
 *
 * A run that returns no plates is a SUCCESSFUL analysis with nothing found,
 * not an error: the copy says so explicitly and offers useful next steps.
 * Every action is a callback owned by the page — no DOM querying.
 */
export default function EmptyState({
  imageName,
  processingTimeSeconds,
  confThreshold,
  onPickSample,
  onLowerThresholdAndRun,
  onReset,
}) {
  return (
    <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-hairline bg-panel p-8 text-center">
      <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-full border border-signal/20 bg-signal/5 text-signal">
        <ScanLine size={24} aria-hidden="true" />
      </div>

      <h3 className="mb-1 font-display text-base font-bold text-ink">
        No license plates detected
      </h3>

      <p className="mb-1 max-w-md text-[12px] leading-relaxed text-ink-dim">
        {imageName ? (
          <>
            <span className="text-ink">{imageName}</span> was processed successfully
            {processingTimeSeconds != null && ` in ${processingTimeSeconds}s`}, but the detector
            found no plate
            {confThreshold != null && (
              <> at the {(confThreshold * 100).toFixed(0)}% confidence threshold</>
            )}
            . Nothing failed — there was simply no readable plate in this scene.
          </>
        ) : (
          'The image was processed successfully, but the detector found no license plate.'
        )}
      </p>

      <ul className="mb-6 max-w-md space-y-1 text-left text-[12px] text-ink-faint">
        <li>· A vehicle may be too far away, blurred, angled away, or cropped out of frame</li>
        <li>· The plate may be obscured by glare, shadow, or another vehicle</li>
        <li>· Lowering the confidence threshold can surface fainter candidates</li>
      </ul>

      <div className="flex flex-wrap items-center justify-center gap-2.5">
        {onPickSample && (
          <button
            type="button"
            onClick={onPickSample}
            className="flex items-center gap-1.5 rounded-md bg-signal px-3.5 py-2 text-[12px] font-semibold text-[#062015] transition-colors hover:bg-signal/90"
          >
            Try a curated sample
            <ArrowRight size={13} aria-hidden="true" />
          </button>
        )}

        {onLowerThresholdAndRun && (
          <button
            type="button"
            onClick={onLowerThresholdAndRun}
            className="flex items-center gap-1.5 rounded-md border border-hairline px-3.5 py-2 text-[12px] font-medium text-ink-dim transition-colors hover:bg-panel-raised hover:text-ink"
          >
            Re-run with a lower threshold
          </button>
        )}

        {onReset && (
          <button
            type="button"
            onClick={onReset}
            className="flex items-center gap-1.5 rounded-md border border-hairline px-3.5 py-2 text-[12px] font-medium text-ink-dim transition-colors hover:bg-panel-raised hover:text-ink"
          >
            <ImagePlus size={13} aria-hidden="true" />
            Analyze another image
          </button>
        )}
      </div>
    </div>
  )
}
