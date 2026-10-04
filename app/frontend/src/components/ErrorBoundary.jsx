import { Component } from 'react'
import { AlertOctagon, RotateCw } from 'lucide-react'

/**
 * Top-level React error boundary.
 *
 * Catches render-time crashes anywhere below it so a single bad component
 * cannot blank the whole app. The user sees a friendly message and a reload
 * action; the stack trace goes to the console for developers only and is
 * never rendered.
 */
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false }
    this.handleReset = this.handleReset.bind(this)
  }

  static getDerivedStateFromError() {
    return { hasError: true }
  }

  componentDidCatch(error, info) {
    // eslint-disable-next-line no-console
    console.error('[ErrorBoundary] render error:', error, info?.componentStack)
  }

  handleReset() {
    this.setState({ hasError: false })
  }

  render() {
    if (!this.state.hasError) return this.props.children

    return (
      <div className="flex min-h-screen items-center justify-center bg-base px-6 py-16 text-center text-ink">
        <div className="w-full max-w-md rounded-lg border border-hairline bg-panel p-8">
          <div className="mx-auto mb-5 flex h-14 w-14 items-center justify-center rounded-full border border-alert/30 bg-alert/10 text-alert">
            <AlertOctagon size={24} aria-hidden="true" />
          </div>

          <h1 className="font-display text-lg font-bold text-ink">
            This page ran into a problem
          </h1>
          <p className="mt-2 text-[13px] leading-relaxed text-ink-dim">
            Something in the interface failed to render. Your uploaded images and detection
            history are unaffected. Reloading usually fixes it.
          </p>

          <div className="mt-6 flex flex-wrap items-center justify-center gap-2">
            <button
              type="button"
              onClick={() => window.location.reload()}
              className="flex items-center gap-1.5 rounded-md bg-signal px-4 py-2 text-[13px] font-semibold text-[#062015] transition-colors hover:bg-signal/90"
            >
              <RotateCw size={13} aria-hidden="true" />
              Reload the app
            </button>
            <button
              type="button"
              onClick={this.handleReset}
              className="rounded-md border border-hairline px-4 py-2 text-[13px] font-medium text-ink-dim transition-colors hover:bg-panel-raised hover:text-ink"
            >
              Try again
            </button>
          </div>
        </div>
      </div>
    )
  }
}