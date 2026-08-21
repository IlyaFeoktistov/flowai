import { useEffect, useState } from 'react'
import { Modal } from '@/shared/ui'
import { plugins } from '../api/api'

export function PluginsModal({ onClose }: { onClose: () => void }) {
  const [report, setReport] = useState<string | null>(null)
  useEffect(() => {
    plugins().then((r) => setReport(r.report))
  }, [])
  return (
    <Modal title="Плагины" onClose={onClose}>
      {report === null ? <p className="dim">Загружаю…</p> : <pre className="report-text">{report}</pre>}
    </Modal>
  )
}
