import { useRef, useState } from 'react'
import { UploadCloud } from 'lucide-react'

/**
 * Drag-and-drop + click-to-upload target.
 *
 * The visible surface is a real <button>, so keyboard activation, Enter/Space
 * and screen-reader semantics all work for free; the hidden file input is
 * only a picker implementation detail. The parent owns validation.
 */
export default function UploadDropzone({ onFile, disabled = false, busy = false }) {
  const inputRef = useRef(null)
  const [isDragging, setIsDragging] = useState(false)
  // dragenter/dragleave fire for every child element; count them so the
  // highlight does not flicker while the pointer moves over the icon.
  const dragDepth = useRef(0)

  function handleDrop(e) {
    e.preventDefault()
    dragDepth.current = 0
    setIsDragging(false)
    if (disabled) return
    const dropped = e.dataTransfer?.files?.[0]
    if (dropped) onFile(dropped)
  }

  function openPicker() {
    if (!disabled) inputRef.current?.click()
  }

  return (
    <div
      onDragEnter={(e) => {
        e.preventDefault()
        dragDepth.current += 1
        if (!disabled) setIsDragging(true)
      }}
      onDragOver={(e) => e.preventDefault()}
      onDragLeave={(e) => {
        e.preventDefault()
        dragDepth.current = Math.max(0, dragDepth.current - 1)
        if (dragDepth.current === 0) setIsDragging(false)
      }}
      onDrop={handleDrop}
      className={`rounded-lg border border-dashed transition-colors ${
        isDragging
          ? 'border-signal bg-signal/5'
          : disabled
            ? 'border-hairline-soft bg-panel opacity-60'
            : 'border-hairline bg-panel hover:border-signal/50 hover:bg-panel-raised/60'
      }`}
    >
      <button
        type="button"
        onClick={openPicker}
        disabled={disabled}
        aria-label="Upload an image for license plate detection. Drag and drop is also supported."
        className="flex w-full cursor-pointer flex-col items-center justify-center px-8 py-20 text-center disabled:cursor-not-allowed"
      >
        <div
          className={`mb-4 flex h-14 w-14 items-center justify-center rounded-full border text-signal transition-colors ${
            isDragging ? 'border-signal bg-signal/10' : 'border-hairline bg-panel-raised'
          }`}
        >
          <UploadCloud size={24} aria-hidden="true" />
        </div>

        <p className="text-[14px] font-semibold text-ink">
          {isDragging
            ? 'Release to load this image'
            : busy
              ? 'Analysis in progress…'
              : 'Drop a vehicle photo here, or click to browse'}
        </p>
        <p className="mt-1 text-[12px] text-ink-faint">
          Supports standard JPEG, PNG, and WebP images up to 10 MB
        </p>
      </button>

      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        className="sr-only"
        aria-hidden="true"
        tabIndex={-1}
        disabled={disabled}
        onChange={(e) => {
          const picked = e.target.files?.[0]
          // Reset the value so re-picking the same file still fires onChange.
          e.target.value = ''
          if (picked) onFile(picked)
        }}
      />
    </div>
  )
}