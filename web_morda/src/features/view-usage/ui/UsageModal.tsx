import { useEffect, useState } from 'react'
import { Modal } from '@/shared/ui'
import { usage } from '../api/api'
import './UsageModal.css'

const LABELS: Record<string, string> = {
  tokens_in: 'Токенов на входе',
  tokens_in_content: 'из них контента',
  tokens_out: 'Токенов на выходе',
  messages: 'Сообщений',
  sessions: 'Сессий',
}

export function UsageModal({ onClose }: { onClose: () => void }) {
  const [totals, setTotals] = useState<Record<string, number> | null>(null)
  useEffect(() => {
    usage().then(setTotals)
  }, [])

  return (
    <Modal title="Использование" onClose={onClose}>
      {!totals ? (
        <p className="dim">Считаю…</p>
      ) : (
        <div className="stat-grid">
          {Object.entries(totals).map(([k, v]) => (
            <div className="stat-tile" key={k}>
              <div className="stat-value">{v.toLocaleString('ru-RU')}</div>
              <div className="stat-label">{LABELS[k] ?? k}</div>
            </div>
          ))}
        </div>
      )}
    </Modal>
  )
}
