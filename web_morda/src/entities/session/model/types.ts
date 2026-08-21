export interface SessionSummary {
  session_id: string
  started_at: string
  last_at: string
  message_count: number
  preview: string
}

// Literal role union duplicated from entities/chat on purpose — importing
// entities/chat's ChatRole here would make entities/chat and entities/session
// depend on each other (chat already depends on session for getSession),
// and FSD entities shouldn't form cycles.
export interface SessionMessage {
  role: 'user' | 'assistant'
  content: string
  ts: string
}
