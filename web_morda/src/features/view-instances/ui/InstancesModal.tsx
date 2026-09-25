import { useEffect, useState } from 'react'
import { Modal } from '@/shared/ui'
import { listInstances, unloadInstance, type ModelInstance } from '../api/api'
import './InstancesModal.css'

function gb(n: number | null, estimated = false): string {
  if (n === null) return '—'
  return `${estimated ? '~' : ''}${(n / 1024 ** 3).toFixed(1)} GB`
}

export function InstancesModal({ onClose }: { onClose: () => void }) {
  const [items, setItems] = useState<ModelInstance[] | null>(null)
  const [errors, setErrors] = useState<string[]>([])
  const [busy, setBusy] = useState<string | null>(null)
  const [status, setStatus] = useState('')

  const refresh = () =>
    listInstances()
      .then((r) => {
        setItems(r.instances)
        setErrors(r.errors)
      })
      .catch((e) => setErrors([String(e)]))

  useEffect(() => {
    refresh()
  }, [])

  const unload = (id: string) => {
    setBusy(id)
    unloadInstance(id)
      .then((r) => setStatus(r.message))
      .catch((e) => setStatus(String(e)))
      .finally(() => {
        setBusy(null)
        refresh()
      })
  }

  return (
    <Modal
      title="Инстансы"
      onClose={onClose}
      footer={
        <button className="btn" onClick={refresh}>
          Обновить
        </button>
      }
    >
      {!items ? (
        <p className="dim">Загружаю…</p>
      ) : (
        <>
          {items.length === 0 && <p className="dim">Сейчас ни одна модель не загружена.</p>}
          <ul className="inst-list">
            {items.map((m) => (
              <li key={m.id}>
                <div className="inst-main">
                  <div className="inst-title">
                    <span className="inst-backend">{m.backend}</span>
                    <b>{m.model}</b>
                  </div>
                  <div className="inst-meta">
                    RAM {gb(m.ram_bytes)} · VRAM {gb(m.vram_bytes, m.vram_estimated)} · {m.details}
                  </div>
                </div>
                <button className="btn btn-danger" disabled={busy !== null} onClick={() => unload(m.id)}>
                  {busy === m.id ? 'Выгружаю…' : 'Выгрузить'}
                </button>
              </li>
            ))}
          </ul>
          {errors.map((e) => (
            <p key={e} className="inst-error">
              {e}
            </p>
          ))}
          {status && <p className="dim">{status}</p>}
        </>
      )}
    </Modal>
  )
}
