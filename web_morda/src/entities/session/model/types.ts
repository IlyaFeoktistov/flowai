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
  // Полная трасса хода (тулы/delegate/thinking/...) — только у assistant-
  // строк, только для ходов ПОСЛЕ появления этой фичи (web/sessions_
  // store.py:save_turn_trace) — старые сессии его не имеют, и entities/
  // chat's buildEntriesFromHistory откатывается для них на плоский рендер.
  // Сырые on_event-payload'ы как есть, минус permission_request/
  // ask_user_request — см. save_turn_trace's докстринг.
  detail?: (Record<string, unknown> & { type: string })[]
}
