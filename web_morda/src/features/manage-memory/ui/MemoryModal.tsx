import { useEffect, useState } from 'react'
import { Modal } from '@/shared/ui'
import { clearAllMemory, deleteFact, deleteKnowledge, getMemory, type MemorySnapshot } from '../api/api'
import './MemoryModal.css'

export function MemoryModal({ onClose }: { onClose: () => void }) {
  const [mem, setMem] = useState<MemorySnapshot | null>(null)
  const refresh = () => getMemory().then(setMem)
  useEffect(() => {
    refresh()
  }, [])

  return (
    <Modal
      title="Память"
      onClose={onClose}
      footer={
        <button
          className="btn btn-danger"
          onClick={() => clearAllMemory().then(refresh)}
          disabled={!mem || (mem.facts.length === 0 && mem.knowledge.length === 0)}
        >
          Удалить всё
        </button>
      }
    >
      {!mem ? (
        <p className="dim">Загружаю…</p>
      ) : mem.facts.length === 0 && mem.knowledge.length === 0 ? (
        <p className="dim">Пока ничего не запомнено.</p>
      ) : (
        <>
          {mem.facts.length > 0 && (
            <section className="mem-section">
              <h3>О пользователе</h3>
              <ul className="mem-list">
                {mem.facts.map((f, i) => (
                  <li key={i}>
                    <span>{f}</span>
                    <button className="icon-btn" onClick={() => deleteFact(i).then(refresh)}>
                      удалить
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          )}
          {mem.knowledge.length > 0 && (
            <section className="mem-section">
              <h3>О проекте</h3>
              <ul className="mem-list">
                {mem.knowledge.map((k) => (
                  <li key={`${k.category}:${k.key}`}>
                    <span>
                      <b>{k.category}</b> · {k.key}: {k.value}
                    </span>
                    <button className="icon-btn" onClick={() => deleteKnowledge(k.category, k.key).then(refresh)}>
                      удалить
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </>
      )}
    </Modal>
  )
}
