import { useCallback, useEffect, useMemo, useState } from 'react'
import './App.css'
import '@/shared/ui/kit.css'
import 'katex/dist/katex.min.css'
import { getProject } from '@/entities/project'
import { listSessions } from '@/entities/session'
import type { SessionSummary } from '@/entities/session'
import { useChatSocket } from '@/entities/chat'
import { BackendLoader, ToastStack } from '@/shared/ui'
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
  const [backendReady, setBackendReady] = useState(false)
  const [projectPath, setProjectPath] = useState('')
  const [homePath, setHomePath] = useState('')
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [showFolderPicker, setShowFolderPicker] = useState(false)
  const [openCommand, setOpenCommand] = useState<CommandKind | null>(null)

  const chat = useChatSocket()

  const refreshSessions = useCallback(() => {
    listSessions().then(setSessions).catch(() => {})
  }, [])

  // При заходе в приложение — сразу пустой новый чат (никакая старая
  // сессия не открывается сама), только сайдбар подтягивает список для
  // ручного выбора/кнопки "Текущая". Раньше тут авто-открывалась самая
  // свежая сессия — от этого отказались: открытие чужого/старого
  // разговора без явного действия пользователя удивляло больше, чем
  // помогало, плюс порождало отдельный класс гонок с "Новый чат".
  //
  // С ретраями, БЕЗ ограничения по числу попыток: make run_web поднимает
  // бэкенд и фронт КОНКУРЕНТНО, без синхронизации — vite обычно готов
  // раньше, чем FastAPI успевает стартовать (замерено ~5.3с холодного
  // старта на разработческой машине — импортирует mcp_agent целиком; ещё
  // дольше при --reload-перезапуске бэкенда посреди уже открытой вкладки).
  // Раньше ретрай был ограничен 20 попытками (~15с) и без единого визуального
  // сигнала — если бэкенд не успевал, список сессий тихо оставался пустым
  // НАВСЕГДА без ретрая дальше, а пользователь тем временем мог уже
  // напечатать и отправить сообщение, которое тут же падало connection
  // refused. Теперь ждём сколько нужно (backendReady=false рисует
  // BackendLoader вместо всего приложения — набрать и отправить запрос,
  // которому нечего слушать, физически нельзя) и ретраим без верхней границы.
  useEffect(() => {
    let cancelled = false

    const loadWithRetry = async () => {
      for (;;) {
        try {
          const [project, list] = await Promise.all([getProject(), listSessions()])
          if (cancelled) return
          setProjectPath(project.path)
          setHomePath(project.home)
          setSessions(list)
          setBackendReady(true)
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

  if (!backendReady) {
    return <BackendLoader />
  }

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
