import { useEffect, useState } from 'react'
import { api } from '../api'
import { Modal } from './Modal'

export type CommandKind = 'doctor' | 'update' | 'clean' | 'usage' | 'memory' | 'plugins' | 'settings'

const TITLES: Record<CommandKind, string> = {
  doctor: 'Доктор',
  update: 'Обновление',
  clean: 'Очистка',
  usage: 'Использование',
  memory: 'Память',
  plugins: 'Плагины',
  settings: 'Настройки',
}

function ReportBody({ text }: { text: string }) {
  return <pre className="report-text">{text}</pre>
}

function DoctorModal({ onClose }: { onClose: () => void }) {
  const [report, setReport] = useState<string | null>(null)
  useEffect(() => {
    api.doctor().then((r) => setReport(r.report))
  }, [])
  return (
    <Modal title={TITLES.doctor} onClose={onClose}>
      {report === null ? <p className="dim">Проверяю Ollama/модели/MCP/хранилище…</p> : <ReportBody text={report} />}
    </Modal>
  )
}

function UpdateModal({ onClose }: { onClose: () => void }) {
  const [report, setReport] = useState<string | null>(null)
  const [running, setRunning] = useState(false)
  const run = () => {
    setRunning(true)
    api
      .update()
      .then((r) => setReport(r.report))
      .finally(() => setRunning(false))
  }
  return (
    <Modal
      title={TITLES.update}
      onClose={onClose}
      footer={
        <button className="btn btn-primary" onClick={run} disabled={running}>
          {running ? 'Проверяю…' : 'Проверить обновления'}
        </button>
      }
    >
      {report ? <ReportBody text={report} /> : <p className="dim">Нажми «Проверить обновления».</p>}
    </Modal>
  )
}

const CLEAN_SCOPES: { key: string; label: string }[] = [
  { key: 'logs', label: 'Логи' },
  { key: 'trash', label: 'Корзина' },
  { key: 'snapshots', label: 'Снимки файлов' },
  { key: 'projects', label: 'Индексы проектов' },
  { key: 'all', label: 'Всё сразу' },
]

function CleanModal({ onClose }: { onClose: () => void }) {
  const [report, setReport] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const refresh = () => api.cleanReport().then((r) => setReport(r.report))
  useEffect(() => {
    refresh()
  }, [])

  const apply = (scope: string) => {
    setBusy(true)
    api
      .cleanApply(scope)
      .then((r) => setReport(r.report))
      .finally(() => setBusy(false))
  }

  return (
    <Modal
      title={TITLES.clean}
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
      {report === null ? <p className="dim">Считаю…</p> : <ReportBody text={report} />}
    </Modal>
  )
}

function UsageModal({ onClose }: { onClose: () => void }) {
  const [totals, setTotals] = useState<Record<string, number> | null>(null)
  useEffect(() => {
    api.usage().then(setTotals)
  }, [])
  const LABELS: Record<string, string> = {
    tokens_in: 'Токенов на входе',
    tokens_in_content: 'из них контента',
    tokens_out: 'Токенов на выходе',
    messages: 'Сообщений',
    sessions: 'Сессий',
  }
  return (
    <Modal title={TITLES.usage} onClose={onClose}>
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

function MemoryModal({ onClose }: { onClose: () => void }) {
  type Mem = { facts: string[]; knowledge: { category: string; key: string; value: string }[] }
  const [mem, setMem] = useState<Mem | null>(null)
  const refresh = () => api.memory().then(setMem)
  useEffect(() => {
    refresh()
  }, [])

  return (
    <Modal
      title={TITLES.memory}
      onClose={onClose}
      footer={
        <button
          className="btn btn-danger"
          onClick={() => api.memoryClearAll().then(refresh)}
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
                    <button className="icon-btn" onClick={() => api.memoryDeleteFact(i).then(refresh)}>
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
                    <button className="icon-btn" onClick={() => api.memoryDeleteKnowledge(k.category, k.key).then(refresh)}>
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

function PluginsModal({ onClose }: { onClose: () => void }) {
  const [report, setReport] = useState<string | null>(null)
  useEffect(() => {
    api.plugins().then((r) => setReport(r.report))
  }, [])
  return <Modal title={TITLES.plugins} onClose={onClose}>{report === null ? <p className="dim">Загружаю…</p> : <ReportBody text={report} />}</Modal>
}

function SettingsModal({ onClose }: { onClose: () => void }) {
  const [settings, setSettings] = useState<Record<string, unknown> | null>(null)
  const refresh = () => api.settingsGet().then(setSettings)
  useEffect(() => {
    refresh()
  }, [])

  const setKey = (key: string, value: unknown) => {
    setSettings((s) => (s ? { ...s, [key]: value } : s))
    api.settingsSet(key, value)
  }

  return (
    <Modal title={TITLES.settings} onClose={onClose}>
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

export function CommandModal({ kind, onClose }: { kind: CommandKind; onClose: () => void }) {
  switch (kind) {
    case 'doctor':
      return <DoctorModal onClose={onClose} />
    case 'update':
      return <UpdateModal onClose={onClose} />
    case 'clean':
      return <CleanModal onClose={onClose} />
    case 'usage':
      return <UsageModal onClose={onClose} />
    case 'memory':
      return <MemoryModal onClose={onClose} />
    case 'plugins':
      return <PluginsModal onClose={onClose} />
    case 'settings':
      return <SettingsModal onClose={onClose} />
  }
}
