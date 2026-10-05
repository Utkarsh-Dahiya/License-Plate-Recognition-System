import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Debounce a fast-changing value (a search box) so the consumer only sees it
 * once the user stops typing. Without this every keystroke reaches the API.
 *
 * Returns `value` unchanged after it has been stable for `delayMs`.
 */
export function useDebouncedValue(value, delayMs = 300) {
  const [settled, setSettled] = useState(value)

  useEffect(() => {
    // Skip the timer entirely when the value is already settled, so the first
    // render does not schedule a redundant state update.
    if (value === settled) return undefined
    const id = setTimeout(() => setSettled(value), delayMs)
    return () => clearTimeout(id)
  }, [value, delayMs, settled])

  return settled
}

/**
 * Load an async resource with the four states every page needs: data,
 * loading, error, and a retry — plus a stale-request guard.
 *
 * The guard is a per-run `cancelled` flag flipped in the effect cleanup. When
 * the deps change (new filter, new page, new search) or the component
 * unmounts, the previous run's callbacks are ignored, so a slow response can
 * never overwrite a newer one and no state update happens after unmount.
 *
 * `fetcher` is read through a ref, so callers may pass an inline closure
 * without re-triggering the request on every render. The request only runs
 * when `deps` change or `retry()` is called — never on mount+StrictMode
 * double-invoke of unrelated renders.
 *
 * `initialData` keeps the previous shape during the very first load when a
 * page renders from a known empty value ({ total: 0, results: [] }).
 *
 * While revalidating, the LAST successful `data` is kept (stale-while-
 * revalidate) instead of flashing an empty screen.
 *
 * `dataKey` is the `deps` value that produced the current `data`, and is
 * cleared to `undefined` when a run FAILS. Keeping the stale payload is only
 * correct while the inputs are unchanged: a consumer keyed by an identifier (a
 * detail drawer, say) must compare `dataKey` and ignore data that belongs to a
 * previous selection or to a failed load, otherwise a failed fetch renders the
 * old record instead of its error.
 */
export function useAsyncResource(fetcher, deps = [], { initialData } = {}) {
  const [data, setData] = useState(initialData)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
  const [attempt, setAttempt] = useState(0)

  // The deps that produced the current `data`, so a consumer can tell whether
  // the payload still belongs to the current selection.
  const [dataKey, setDataKey] = useState(deps[0])

  const fetcherRef = useRef(fetcher)
  useEffect(() => {
    fetcherRef.current = fetcher
  })

  useEffect(() => {
    let cancelled = false
    // Captured per run so the success handler records which selection the
    // payload belongs to.
    const runKey = deps[0]
    setLoading(true)
    setError(null)

    Promise.resolve()
      .then(() => fetcherRef.current())
      .then(
        (result) => {
          if (cancelled) return
          setData(result)
          setDataKey(runKey)
          setError(null)
          setLoading(false)
        },
        (err) => {
          if (cancelled) return
          // The retained payload no longer represents the current inputs: a
          // failed run must never be shown as if it had succeeded. Clearing the
          // key makes keyed consumers fall back to the error branch.
          setDataKey(undefined)
          setError(err || new Error('The request failed.'))
          setLoading(false)
        },
      )

    // Invalidates this run: a newer run started, or the page unmounted.
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, attempt])

  const retry = useCallback(() => setAttempt((n) => n + 1), [])

  return { data, error, loading, retry, setData, dataKey }
}
