import { Routes, Route } from 'react-router-dom'
import AppLayout from './layouts/AppLayout.jsx'
import Overview from './pages/Overview.jsx'
import Detection from './pages/Detection.jsx'
import ImageAnalysis from './pages/ImageAnalysis.jsx'
import BatchIntelligence from './pages/BatchIntelligence.jsx'
import VideoAnalytics from './pages/VideoAnalytics.jsx'
import PlateRegistry from './pages/PlateRegistry.jsx'
import ModelPerformance from './pages/ModelPerformance.jsx'
import System from './pages/System.jsx'

export default function App() {
  return (
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
      </Route>
    </Routes>
  )
}
