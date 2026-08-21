import { useCallback, useEffect, useState } from 'react'
import './App.css'
import { api } from './api'
import { useChatSocket } from './useChatSocket'
import { Sidebar } from './components/Sidebar'
import { Chat } from './components/Chat'
import { InputBar } from './components/InputBar'
import { FolderPicker } from './components/FolderPicker'
import { CommandModal, type CommandKind } from './components/CommandModals'
import type { SessionSummary } from './types'

function App() {
  const [projectPath, setProjectPath] = useState('')
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [showFolderPicker, setShowFolderPicker] = useState(false)
  const [openCommand, setOpenCommand] = useState<CommandKind | null>(null)

  const chat = useChatSocket()

  const refreshSessions = useCallback(() => {
    api.listSessions().then(setSessions).catch(() => {})
  }, [])

  useEffect(() => {
    api.getProject().then((r) => setProjectPath(r.path))
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
        onOpenCommand={setOpenCommand}
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

      {openCommand && <CommandModal kind={openCommand} onClose={() => setOpenCommand(null)} />}
    </div>
  )
}

export default App
