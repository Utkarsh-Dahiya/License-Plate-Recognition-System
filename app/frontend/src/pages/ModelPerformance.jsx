import { useEffect, useState } from 'react'
import Header from '../components/Header.jsx'
import { Panel } from '../components/Panel.jsx'
import { api } from '../lib/api'

function Row({ label, value }) {
  return (
    <div className="flex items-center justify-between border-b border-hairline-soft py-2.5 text-[12px] last:border-0">
      <span className="text-ink-dim">{label}</span>
      <span className="font-mono text-ink">{value ?? 'N/A'}</span>
    </div>
  )
}

function metricDisplay(value) {
  if (value === null || value === undefined || value === '') return 'N/A'
  return String(value)
}

export default function ModelPerformance() {
  const [model, setModel] = useState(null)

  useEffect(() => {
    api
      .model()
      .then(setModel)
      .catch(() => setModel(null))
  }, [])

  const tm = model?.training_metrics

  return (
    <div>
      <Header
        title="Model Performance"
        description="What the models are, and what's actually been measured — no invented accuracy figures."
      />

      <div className="grid grid-cols-1 gap-4 px-8 py-6 lg:grid-cols-2">
        <Panel title="Detector">
          <Row label="Model" value={model?.detector.name} />
          <Row label="Framework" value={model?.detector.framework} />
          <Row label="Weights found" value={model?.detector.weights_found ? 'Yes' : 'No'} />
        </Panel>

        <Panel title="OCR">
          <Row label="Engine" value={model?.ocr.engine} />
          <Row label="Languages" value={model?.ocr.languages?.join(', ')} />
        </Panel>

        <Panel
          title="Training metrics"
          subtitle={tm?.note || 'YOLO detection validation from results.csv'}
        >
          <Row label="Epoch" value={metricDisplay(tm?.epoch)} />
          <Row label="Precision" value={metricDisplay(tm?.precision)} />
          <Row label="Recall" value={metricDisplay(tm?.recall)} />
          <Row label="mAP50" value={metricDisplay(tm?.mAP50 ?? tm?.mAP)} />
          <Row label="mAP50-95" value={metricDisplay(tm?.mAP50_95)} />
        </Panel>

        <Panel title="Observed batch performance" subtitle="Based on 658 processed images — not a held-out validation set. OCR rate is text extraction, not plate accuracy.">
          <Row
            label="Mean YOLO confidence"
            value={model && `${model.observed_performance.mean_yolo_confidence}%`}
          />
          <Row
            label="Mean OCR confidence"
            value={model && `${model.observed_performance.mean_ocr_confidence}%`}
          />
          <Row
            label="OCR text extracted"
            value={model && `${model.observed_performance.ocr_success_rate}%`}
          />
        </Panel>

        <Panel title="Inference pipeline" className="lg:col-span-2">
          <div className="flex flex-wrap items-center gap-2">
            {model?.pipeline.map((step, i) => (
              <div key={step} className="flex items-center gap-2">
                <span className="rounded border border-hairline-soft px-2.5 py-1 font-mono text-[11px] text-ink-dim">
                  {step}
                </span>
                {i < model.pipeline.length - 1 && <span className="text-ink-faint">→</span>}
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </div>
  )
}
