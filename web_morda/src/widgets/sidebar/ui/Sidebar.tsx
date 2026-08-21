import type { SessionSummary } from '@/entities/session'
import { IconFolder, IconPlus, IconChevronRight } from '@/shared/ui'
import './Sidebar.css'

function shortenPath(path: string, home: string): string {
  if (home && (path === home || path.startsWith(home + '/'))) {
    return '~' + path.slice(home.length)
  }
  return path
}

function timeLabel(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  const now = new Date()
  const sameDay = d.toDateString() === now.toDateString()
  return sameDay
    ? d.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' })
    : d.toLocaleDateString('ru-RU', { day: '2-digit', month: '2-digit' })
}

export function Sidebar({
  projectPath,
  homePath,
  onPickFolder,
  commands,
  onOpenCommand,
  onNewChat,
  sessions,
  activeSessionId,
  onOpenSession,
}: {
  projectPath: string
  homePath: string
  onPickFolder: () => void
  commands: { key: string; label: string }[]
  onOpenCommand: (key: string) => void
  onNewChat: () => void
  sessions: SessionSummary[]
  activeSessionId: string | null
  onOpenSession: (id: string) => void
}) {
  // "Текущая" — не скролл к уже открытой сессии, а переход к самой свежей
  // (sessions[0], список отсортирован по last_at DESC на бэкенде, см.
  // web/sessions_store.py) — на случай если открыта старая сессия из
  // списка или пустой новый чат, а хочется вернуться к тому, с чем
  // реально сейчас работаешь.
  const mostRecentId = sessions[0]?.session_id
  const goToCurrent = () => {
    if (mostRecentId) onOpenSession(mostRecentId)
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <span className="brand-mark" aria-hidden="true">F</span>
        FlowAI
      </div>

      <button className="folder-btn" onClick={onPickFolder} title={projectPath}>
        <IconFolder />
        <span className="folder-btn-path">{shortenPath(projectPath, homePath)}</span>
      </button>

      <button className="new-chat-btn" onClick={onNewChat}>
        <IconPlus /> Новый чат
      </button>

      <div className="sidebar-divider" />

      <nav className="command-nav">
        {commands.map((c) => (
          <button key={c.key} className="command-nav-item" onClick={() => onOpenCommand(c.key)}>
            {c.label}
          </button>
        ))}
      </nav>

      <div className="sidebar-divider" />

      <div className="session-list-header">
        <span className="session-list-label">Сессии</span>
        {mostRecentId && mostRecentId !== activeSessionId && (
          <button className="goto-current-btn" onClick={goToCurrent} title="Перейти к текущей сессии">
            Текущая <IconChevronRight />
          </button>
        )}
      </div>
      <ul className="session-list">
        {sessions.map((s) => (
          <li key={s.session_id}>
            <button
              className={'session-item' + (s.session_id === activeSessionId ? ' active' : '')}
              onClick={() => onOpenSession(s.session_id)}
            >
              <div className="session-preview">{s.preview || '(пусто)'}</div>
              <div className="session-meta">
                {timeLabel(s.last_at)} · {s.message_count}
              </div>
            </button>
          </li>
        ))}
        {sessions.length === 0 && <li className="dim session-empty">Пока пусто</li>}
      </ul>
    </aside>
  )
}
