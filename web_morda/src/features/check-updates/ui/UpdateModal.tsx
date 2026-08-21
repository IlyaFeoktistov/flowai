import { useState } from 'react'
import { Modal } from '@/shared/ui'
import { checkUpdates } from '../api/api'

export function UpdateModal({ onClose }: { onClose: () => void }) {
  const [report, setReport] = useState<string | null>(null)
  const [running, setRunning] = useState(false)

  const run = () => {
    setRunning(true)
    checkUpdates()
      .then((r) => setReport(r.report))
      .finally(() => setRunning(false))
  }

  return (
    <Modal
      title="Обновление"
      onClose={onClose}
      footer={
        <button className="btn btn-primary" onClick={run} disabled={running}>
          {running ? 'Проверяю…' : 'Проверить обновления'}
        </button>
      }
    >
      {report ? <pre className="report-text">{report}</pre> : <p className="dim">Нажми «Проверить обновления».</p>}
    </Modal>
  )
}
