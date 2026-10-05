import { lazy, Suspense } from 'react'
import { Routes, Route } from 'react-router-dom'
import AppLayout from './layouts/AppLayout.jsx'
import ErrorBoundary from './components/ErrorBoundary.jsx'
import PageFallback from './components/PageFallback.jsx'

/*
 * Route-level code splitting (Phase 4A).
 *
 * Every page is a separate chunk fetched on navigation, so the initial load
 * no longer pays for pages the visitor never opens. The important win is
 * `recharts` (~400 kB): it is used ONLY by Overview, and because Overview is
 * now lazy, the charting code is no longer part of the startup payload.
 *
 * `react-vendor` / `charts-vendor` stay as static chunks (see vite.config.js)
 * so the charting library is cached separately once Overview is opened.
 *
 * This changes only WHEN a module is fetched, never WHAT it renders: each
 * lazy component is the same component with the same props and behaviour.
 */
const Overview = lazy(() => import('./pages/Overview.jsx'))
const Detection = lazy(() => import('./pages/Detection.jsx'))
const ImageAnalysis = lazy(() => import('./pages/ImageAnalysis.jsx'))
const BatchIntelligence = lazy(() => import('./pages/BatchIntelligence.jsx'))
const VideoAnalytics = lazy(() => import('./pages/VideoAnalytics.jsx'))
const PlateRegistry = lazy(() => import('./pages/PlateRegistry.jsx'))
const ModelPerformance = lazy(() => import('./pages/ModelPerformance.jsx'))
const System = lazy(() => import('./pages/System.jsx'))
const NotFound = lazy(() => import('./pages/NotFound.jsx'))

export default function App() {
  return (
    <ErrorBoundary>
      <Suspense fallback={<PageFallback />}>
        <Routes>
          <Route element={<AppLayout />}>
            <Route path="/" element={<Overview />} />
            <Route path="/detection" element={<Detection />} />
            <Route path="/image-analysis" element={<ImageAnalysis />} />
            <Route path="/batch" element={<BatchIntelligence />} />
            <Route path="/video" element={<VideoAnalytics />} />
            <Route path="/registry" element={<PlateRegistry />} />
            <Route path="/model" element={<ModelPerformance />} />
            <Route path="/system" element={<System />} />
            {/* Catch-all: unknown URLs render the 404 page inside the app shell. */}
            <Route path="*" element={<NotFound />} />
          </Route>
        </Routes>
      </Suspense>
    </ErrorBoundary>
  )
}
