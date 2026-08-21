import { useCallback, useEffect, useState } from 'react'
import './App.css'
import '@/shared/ui/kit.css'
import { getProject } from '@/entities/project'
import { listSessions } from '@/entities/session'
import type { SessionSummary } from '@/entities/session'
import { useChatSocket } from '@/entities/chat'
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
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [showFolderPicker, setShowFolderPicker] = useState(false)
  const [openCommand, setOpenCommand] = useState<CommandKind | null>(null)

  const chat = useChatSocket()

  const refreshSessions = useCallback(() => {
    listSessions().then(setSessions).catch(() => {})
  }, [])

  useEffect(() => {
    getProject().then((r) => setProjectPath(r.path))
    refreshSessions()
  }, [refreshSessions])

  // Новая запись в списке слева появляется только после первой реплики —
  // подтягиваем список при каждом завершении хода, а не только при старте.
  useEffect(() => {
    if (!chat.isStreaming) refreshSessions()
  }, [chat.isStreaming, refreshSessions])

  return (
    <div className="app">
      <Sidebar
        projectPath={projectPath}
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
        {chat.error && <div className="banner-error">{chat.error}</div>}
        <InputBar disabled={chat.isStreaming} onSend={chat.sendMessage} />
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
