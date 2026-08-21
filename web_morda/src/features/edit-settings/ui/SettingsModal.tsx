import { useEffect, useState } from 'react'
import { Modal } from '@/shared/ui'
import { getModels, getSettings, setSetting } from '../api/api'
import './SettingsModal.css'

// Те же три ключа, что в терминале помечены типом "ollama_model"
// (ui/tui/settings.py:_ITEMS) — выбор из уже установленных моделей, а не
// произвольный текст.
const MODEL_KEYS = new Set(['chat_model', 'vision_model', 'voice_chat_model'])

export function SettingsModal({ onClose }: { onClose: () => void }) {
  const [settings, setSettings] = useState<Record<string, unknown> | null>(null)
  const [models, setModels] = useState<string[]>([])
  const refresh = () => getSettings().then(setSettings)
  useEffect(() => {
    refresh()
    getModels()
      .then((r) => setModels(r.models))
      .catch(() => {})
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
              ) : MODEL_KEYS.has(key) && typeof value === 'string' ? (
                <select className="settings-input" value={value} onChange={(e) => setKey(key, e.target.value)}>
                  {/* текущее значение может быть моделью, которой уже нет в
                      `ollama list` (переименована/удалена) — не терять его молча */}
                  {!models.includes(value) && <option value={value}>{value} (не установлена)</option>}
                  {models.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
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
