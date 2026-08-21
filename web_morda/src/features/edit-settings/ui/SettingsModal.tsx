import { useEffect, useState } from 'react'
import { Modal } from '@/shared/ui'
import { getSettings, setSetting } from '../api/api'
import './SettingsModal.css'

export function SettingsModal({ onClose }: { onClose: () => void }) {
  const [settings, setSettings] = useState<Record<string, unknown> | null>(null)
  const refresh = () => getSettings().then(setSettings)
  useEffect(() => {
    refresh()
  }, [])

  const setKey = (key: string, value: unknown) => {
    setSettings((s) => (s ? { ...s, [key]: value } : s))
    setSetting(key, value)
  }

  return (
    <Modal title="Настройки" onClose={onClose}>
      {!settings ? (
        <p className="dim">Загружаю…</p>
      ) : (
        <ul className="settings-list">
          {Object.entries(settings).map(([key, value]) => (
            <li key={key}>
              <span className="settings-key">{key}</span>
              {typeof value === 'boolean' ? (
                <label className="switch">
                  <input type="checkbox" checked={value} onChange={(e) => setKey(key, e.target.checked)} />
                  <span className="switch-track" />
                </label>
              ) : typeof value === 'number' ? (
                <input
                  className="settings-input"
                  type="number"
                  defaultValue={value}
                  onBlur={(e) => setKey(key, Number(e.target.value))}
                />
              ) : (
                <input
                  className="settings-input"
                  type="text"
                  defaultValue={String(value)}
                  onBlur={(e) => setKey(key, e.target.value)}
                />
              )}
            </li>
          ))}
        </ul>
      )}
    </Modal>
  )
}
