export type ChatRole = 'user' | 'assistant'

export interface AskUserOption {
  label: string
  description?: string
}

export interface ToolChild {
  id: string
  name: string
  args: unknown
  result?: string
  status: 'running' | 'done'
}

// One item inside a live/just-completed turn's detail trace — everything
// the terminal TUI (ui/stream.py) would have printed for this turn, minus
// the parts that only matter live (footer/spinner phrases).
export type TurnItem =
  | { kind: 'stage'; id: string; stage: string }
  | { kind: 'text'; id: string; text: string; open: boolean }
  | { kind: 'thinking'; id: string; text: string; open: boolean }
  | {
      kind: 'tool'
      id: string
      name: string
      args: unknown
      result?: string
      diff?: string
      status: 'running' | 'done'
      // delegate → X sub-calls (delegate_tool.py) nest here instead of
      // showing up as sibling tool cards — only ever populated when
      // name === 'delegate'.
      children?: ToolChild[]
    }
  | { kind: 'plan'; id: string; steps: string[]; doneIndexes: number[]; currentIndex: number | null }
  | {
      kind: 'permission'
      id: string
      action: string
      detail: string
      resolved?: 'y' | 'a' | 'n'
    }
  | {
      kind: 'ask_user'
      id: string
      question: string
      options: AskUserOption[]
      recommended?: string | null
      resolved?: string
    }
  | { kind: 'error'; id: string; message: string }
  // Сообщение, подложенное ПОКА этот же ход уже шёл (mid_turn_injected из
  // agent.py) — не отдельный ход, а аннотация внутри текущего.
  | { kind: 'mid_turn'; id: string; text: string }
  | { kind: 'stopped'; id: string }

export interface Turn {
  kind: 'turn'
  id: string
  userText: string
  items: TurnItem[]
  complete: boolean
}

export interface HistoryMessage {
  kind: 'message'
  id: string
  role: ChatRole
  content: string
}

export type ConversationEntry = HistoryMessage | Turn

export type ConnectionStatus = 'idle' | 'connecting' | 'open' | 'closed'
