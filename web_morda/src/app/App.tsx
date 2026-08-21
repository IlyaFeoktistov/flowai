import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import '@/shared/ui/kit.css'
import 'katex/dist/katex.min.css'
import { getProject } from '@/entities/project'
import { listSessions } from '@/entities/session'
import type { SessionSummary } from '@/entities/session'
import { useChatSocket } from '@/entities/chat'
import { ToastStack } from '@/shared/ui'
import { Sidebar } from '@/widgets/sidebar'
import { Chat } from '@/widgets/chat-panel'
import { InputBar } from '@/features/send-message'
import { FolderPicker } from '@/features/pick-folder'
import { DoctorModal } from '@/features/view-doctor'
import { UpdateModal } from '@/features/check-updates'
import { CleanModal } from '@/features/clean-storage'
import { UsageModal } from '@/features/view-usage'
import { MemoryModal } from '@/features/manage-memory'
import { PluginsModal } from '@/features/view-plugins'
import { SettingsModal } from '@/features/edit-settings'

type CommandKind = 'doctor' | 'update' | 'clean' | 'usage' | 'memory' | 'plugins' | 'settings'

const COMMANDS: { key: CommandKind; label: string }[] = [
  { key: 'doctor', label: 'Доктор' },
  { key: 'update', label: 'Обновления' },
  { key: 'clean', label: 'Очистка' },
  { key: 'usage', label: 'Использование' },
  { key: 'memory', label: 'Память' },
  { key: 'plugins', label: 'Плагины' },
  { key: 'settings', label: 'Настройки' },
]

function App() {
  const [projectPath, setProjectPath] = useState('')
  const [homePath, setHomePath] = useState('')
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [showFolderPicker, setShowFolderPicker] = useState(false)
  const [openCommand, setOpenCommand] = useState<CommandKind | null>(null)

  const chat = useChatSocket()

  const refreshSessions = useCallback(() => {
    listSessions().then(setSessions).catch(() => {})
  }, [])

  // Открываем самую свежую сессию сразу при заходе в приложение — раньше
  // WS-соединение вообще не поднималось, пока не отправишь первое
  // сообщение, так что список сессий слева был виден, а сама последняя
  // переписка — нет, пока её явно не открыть кликом. Только один раз за
  // время жизни компонента (didAutoOpenRef) — иначе каждый повторный
  // refreshSessions() (после каждого хода, см. ниже) выдёргивал бы
  // пользователя обратно в последнюю сессию, даже если он сейчас смотрит
  // другую.
  //
  // С ретраями: make run_web поднимает бэкенд и фронт КОНКУРЕНТНО, без
  // синхронизации — vite обычно готов раньше, чем FastAPI успевает
  // стартовать (или посреди --reload-перезапуска после правки бэкенда).
  // Без ретрая первый неудачный fetch тут просто тихо проглатывался
  // (.catch(() => {})) и список сессий оставался пустым НАВСЕГДА, пока
  // страницу не перезагрузишь руками.
  const didAutoOpenRef = useRef(false)
  useEffect(() => {
    let cancelled = false

    const loadWithRetry = async () => {
      // 20 попыток * 750мс = 15с — с запасом покрывает холодный старт
      // бэкенда (~5.3с на этой машине: FastAPI импортирует mcp_agent целиком
      // при первом старте процесса, не мгновенно).
      for (let attempt = 0; attempt < 20; attempt++) {
        try {
          const [project, list] = await Promise.all([getProject(), listSessions()])
          if (cancelled) return
          setProjectPath(project.path)
          setHomePath(project.home)
          setSessions(list)
          if (!didAutoOpenRef.current && list.length > 0) {
            didAutoOpenRef.current = true
            chat.openSession(list[0].session_id)
          }
          return
        } catch {
          if (cancelled) return
          await new Promise((r) => setTimeout(r, 750))
        }
      }
    }

    loadWithRetry()
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Новая запись в списке слева появляется только после первой реплики —
  // подтягиваем список при каждом завершении хода, а не только при старте.
  useEffect(() => {
    if (!chat.isStreaming) refreshSessions()
  }, [chat.isStreaming, refreshSessions])

  // Для войс-режима (InputBar's TTS-озвучка ответа) — текст последнего
  // ответа ассистента, откуда бы он ни пришёл (текущий ход или уже
  // сохранённая история).
  const lastAnswerText = useMemo(() => {
    for (let i = chat.entries.length - 1; i >= 0; i--) {
      const entry = chat.entries[i]
      if (entry.kind === 'message') {
        return entry.role === 'assistant' ? entry.content : null
      }
      const textItems = entry.items.filter((it) => it.kind === 'text')
      const last = textItems[textItems.length - 1]
      return last && last.kind === 'text' ? last.text : null
    }
    return null
  }, [chat.entries])

  return (
    <div className="app">
      <ToastStack toasts={chat.errors} onDismiss={chat.dismissError} />
      <Sidebar
        projectPath={projectPath}
        homePath={homePath}
        onPickFolder={() => setShowFolderPicker(true)}
        commands={COMMANDS}
        onOpenCommand={(key) => setOpenCommand(key as CommandKind)}
        onNewChat={chat.startNewChat}
        sessions={sessions}
        activeSessionId={chat.sessionId}
        onOpenSession={chat.openSession}
      />

      <main className="chat-panel">
        <Chat entries={chat.entries} onRespondPermission={chat.respondPermission} onRespondAskUser={chat.respondAskUser} />
        <InputBar
          streaming={chat.isStreaming}
          pendingCount={chat.pendingCount}
          lastAnswerText={lastAnswerText}
          onSend={chat.sendMessage}
          onStop={chat.stopCurrentTurn}
        />
      </main>

      {showFolderPicker && (
        <FolderPicker
          startPath={projectPath}
          onClose={() => setShowFolderPicker(false)}
          onChosen={(p) => {
            setProjectPath(p)
            setShowFolderPicker(false)
          }}
        />
      )}

      {openCommand === 'doctor' && <DoctorModal onClose={() => setOpenCommand(null)} />}
      {openCommand === 'update' && <UpdateModal onClose={() => setOpenCommand(null)} />}
      {openCommand === 'clean' && <CleanModal onClose={() => setOpenCommand(null)} />}
      {openCommand === 'usage' && <UsageModal onClose={() => setOpenCommand(null)} />}
      {openCommand === 'memory' && <MemoryModal onClose={() => setOpenCommand(null)} />}
      {openCommand === 'plugins' && <PluginsModal onClose={() => setOpenCommand(null)} />}
      {openCommand === 'settings' && <SettingsModal onClose={() => setOpenCommand(null)} />}
    </div>
  )
}

export default App
