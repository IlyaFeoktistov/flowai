import { useEffect, useState } from 'react'
import { Modal } from '@/shared/ui'
import { cleanApply, cleanReport } from '../api/api'

const CLEAN_SCOPES: { key: string; label: string }[] = [
  { key: 'logs', label: 'Логи' },
  { key: 'trash', label: 'Корзина' },
  { key: 'snapshots', label: 'Снимки файлов' },
  { key: 'projects', label: 'Индексы проектов' },
  { key: 'all', label: 'Всё сразу' },
]

export function CleanModal({ onClose }: { onClose: () => void }) {
  const [report, setReport] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const refresh = () => cleanReport().then((r) => setReport(r.report))
  useEffect(() => {
    refresh()
  }, [])

  const apply = (scope: string) => {
    setBusy(true)
    cleanApply(scope)
      .then((r) => setReport(r.report))
      .finally(() => setBusy(false))
  }

  return (
    <Modal
      title="Очистка"
      onClose={onClose}
      footer={
        <div className="btn-row">
          {CLEAN_SCOPES.map((s) => (
            <button key={s.key} className="btn" disabled={busy} onClick={() => apply(s.key)}>
              {s.label}
            </button>
          ))}
        </div>
      }
    >
      {report === null ? <p className="dim">Считаю…</p> : <pre className="report-text">{report}</pre>}
    </Modal>
  )
}
