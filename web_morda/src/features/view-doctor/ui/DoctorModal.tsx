import { useEffect, useState } from 'react'
import { Modal } from '@/shared/ui'
import { doctor } from '../api/api'

export function DoctorModal({ onClose }: { onClose: () => void }) {
  const [report, setReport] = useState<string | null>(null)
  useEffect(() => {
    doctor().then((r) => setReport(r.report))
  }, [])
  return (
    <Modal title="Доктор" onClose={onClose}>
      {report === null ? (
        <p className="dim">Проверяю Ollama/модели/MCP/хранилище…</p>
      ) : (
        <pre className="report-text">{report}</pre>
      )}
    </Modal>
  )
}
