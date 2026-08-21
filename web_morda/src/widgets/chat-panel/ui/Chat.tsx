import { useEffect, useRef } from 'react'
import type { ConversationEntry } from '@/entities/chat'
import { renderMarkdown } from '@/shared/lib'
import { Collapsible } from '@/shared/ui'
import { TurnView } from './TurnView'
import './Chat.css'

export function Chat({
  entries,
  onRespondPermission,
  onRespondAskUser,
}: {
  entries: ConversationEntry[]
  onRespondPermission: (id: string, answer: 'y' | 'a' | 'n') => void
  onRespondAskUser: (id: string, answer: string) => void
}) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: 'end' })
  }, [entries])

  if (entries.length === 0) {
    return (
      <div className="chat-empty">
        <span className="brand-mark brand-mark-lg" aria-hidden="true">F</span>
        <p>Спроси что-нибудь, или выбери сессию слева.</p>
      </div>
    )
  }

  return (
    <div className="chat-scroll">
      <div className="chat-inner">
        {entries.map((entry) =>
          entry.kind === 'message' ? (
            <div className={`msg msg-${entry.role}`} key={entry.id}>
              <div className="msg-bubble">
                <Collapsible>{entry.role === 'assistant' ? renderMarkdown(entry.content) : entry.content}</Collapsible>
              </div>
            </div>
          ) : (
            <TurnView
              key={entry.id}
              turn={entry}
              onRespondPermission={onRespondPermission}
              onRespondAskUser={onRespondAskUser}
            />
          ),
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
