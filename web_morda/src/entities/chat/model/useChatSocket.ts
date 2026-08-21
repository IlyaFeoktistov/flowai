import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { getSession } from '@/entities/session'
import type { AskUserOption, ConnectionStatus, ConversationEntry, Turn, TurnItem } from './types'

let idSeed = 0
const nextId = () => `${Date.now().toString(36)}-${idSeed++}`

function mapTurnItem(entries: ConversationEntry[], turnId: string, fn: (t: Turn) => Turn): ConversationEntry[] {
  return entries.map((e) => (e.kind === 'turn' && e.id === turnId ? fn(e) : e))
}

function pushItem(turn: Turn, item: TurnItem): Turn {
  return { ...turn, items: [...turn.items, item] }
}

function updateItem(turn: Turn, itemId: string, fn: (i: TurnItem) => TurnItem): Turn {
  return { ...turn, items: turn.items.map((i) => (i.id === itemId ? fn(i) : i)) }
}

function updateLastOfKind<K extends TurnItem['kind']>(
  turn: Turn,
  kind: K,
  fn: (i: Extract<TurnItem, { kind: K }>) => TurnItem,
): Turn {
  for (let i = turn.items.length - 1; i >= 0; i--) {
    if (turn.items[i].kind === kind) {
      const items = [...turn.items]
      items[i] = fn(items[i] as Extract<TurnItem, { kind: K }>)
      return { ...turn, items }
    }
  }
  return turn
}

function wsUrl(sessionId: string | null): string {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const base = `${proto}//${window.location.host}/api/v1/ws/chat`
  return sessionId ? `${base}?session_id=${encodeURIComponent(sessionId)}` : base
}

export function useChatSocket() {
  const [entries, setEntries] = useState<ConversationEntry[]>([])
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [status, setStatus] = useState<ConnectionStatus>('idle')
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const currentTurnIdRef = useRef<string | null>(null)
  const pendingTextRef = useRef<string | null>(null)
  const pendingThinkingRef = useRef<string | null>(null)
  const pendingSendRef = useRef<string | null>(null)

  const closeSocket = useCallback(() => {
    wsRef.current?.close()
    wsRef.current = null
  }, [])

  const connect = useCallback((withSessionId: string | null) => {
    closeSocket()
    setStatus('connecting')
    setError(null)
    const ws = new WebSocket(wsUrl(withSessionId))
    wsRef.current = ws

    ws.onopen = () => setStatus('open')
    ws.onclose = () => {
      setStatus('closed')
      setIsStreaming(false)
    }
    ws.onerror = () => setError('Соединение с агентом прервалось')

    ws.onmessage = (ev) => {
      const event = JSON.parse(ev.data) as Record<string, unknown> & { type: string }
      handleEvent(event)
    }

    function handleEvent(event: Record<string, unknown> & { type: string }) {
      if (event.type === 'session_started') {
        const id = event.session_id as string
        setSessionId(id)
        if (pendingSendRef.current) {
          const text = pendingSendRef.current
          pendingSendRef.current = null
          sendRaw(text)
        }
        return
      }

      const turnId = currentTurnIdRef.current
      if (!turnId) return
      setEntries((prev) => {
        switch (event.type) {
          case 'stage_changed':
            return mapTurnItem(prev, turnId, (t) =>
              pushItem(t, { kind: 'stage', id: nextId(), stage: event.stage as string }),
            )

          case 'thinking_start': {
            const id = nextId()
            pendingThinkingRef.current = id
            return mapTurnItem(prev, turnId, (t) => pushItem(t, { kind: 'thinking', id, text: '', open: true }))
          }
          case 'thinking_chunk': {
            const id = pendingThinkingRef.current
            if (!id) return prev
            return mapTurnItem(prev, turnId, (t) =>
              updateItem(t, id, (i) => (i.kind === 'thinking' ? { ...i, text: i.text + (event.text as string) } : i)),
            )
          }
          case 'thinking_end': {
            const id = pendingThinkingRef.current
            pendingThinkingRef.current = null
            if (!id) return prev
            return mapTurnItem(prev, turnId, (t) =>
              updateItem(t, id, (i) => (i.kind === 'thinking' ? { ...i, open: false } : i)),
            )
          }

          case 'answer_start': {
            const id = nextId()
            pendingTextRef.current = id
            return mapTurnItem(prev, turnId, (t) => pushItem(t, { kind: 'text', id, text: '', open: true }))
          }
          case 'answer_chunk': {
            const id = pendingTextRef.current
            if (!id) return prev
            return mapTurnItem(prev, turnId, (t) =>
              updateItem(t, id, (i) => (i.kind === 'text' ? { ...i, text: i.text + (event.text as string) } : i)),
            )
          }
          case 'answer_end': {
            const id = pendingTextRef.current
            pendingTextRef.current = null
            if (!id) return prev
            return mapTurnItem(prev, turnId, (t) => updateItem(t, id, (i) => (i.kind === 'text' ? { ...i, open: false } : i)))
          }

          case 'tool_start':
            return mapTurnItem(prev, turnId, (t) =>
              pushItem(t, {
                kind: 'tool',
                id: event.id as string,
                name: event.name as string,
                args: event.args,
                status: 'running',
              }),
            )
          case 'tool_end':
            return mapTurnItem(prev, turnId, (t) =>
              updateItem(t, event.id as string, (i) =>
                i.kind === 'tool'
                  ? { ...i, status: 'done', result: event.result as string, diff: event.diff as string | undefined }
                  : i,
              ),
            )

          case 'plan_steps':
            return mapTurnItem(prev, turnId, (t) =>
              pushItem(t, {
                kind: 'plan',
                id: nextId(),
                steps: event.steps as string[],
                doneIndexes: [],
                currentIndex: null,
              }),
            )
          case 'plan_step_done':
            return mapTurnItem(prev, turnId, (t) =>
              updateLastOfKind(t, 'plan', (i) => ({
                ...i,
                doneIndexes: [...i.doneIndexes, event.index as number],
              })),
            )

          case 'permission_request':
            return mapTurnItem(prev, turnId, (t) =>
              pushItem(t, {
                kind: 'permission',
                id: event.id as string,
                action: event.action as string,
                detail: event.detail as string,
              }),
            )
          case 'ask_user_request':
            return mapTurnItem(prev, turnId, (t) =>
              pushItem(t, {
                kind: 'ask_user',
                id: event.id as string,
                question: event.question as string,
                options: (event.options as AskUserOption[]) ?? [],
                recommended: event.recommended as string | null,
              }),
            )

          case 'error':
            setError(event.message as string)
            return mapTurnItem(prev, turnId, (t) =>
              pushItem(t, { kind: 'error', id: nextId(), message: event.message as string }),
            )

          case 'turn_complete':
            currentTurnIdRef.current = null
            setIsStreaming(false)
            return mapTurnItem(prev, turnId, (t) => ({ ...t, complete: true }))

          default:
            return prev
        }
      })
    }

    function sendRaw(text: string) {
      ws.send(JSON.stringify({ type: 'user_message', text }))
    }
  }, [closeSocket])

  const sendMessage = useCallback(
    (text: string) => {
      const trimmed = text.trim()
      if (!trimmed || isStreaming) return

      const turnId = nextId()
      currentTurnIdRef.current = turnId
      setIsStreaming(true)
      setEntries((prev) => [...prev, { kind: 'turn', id: turnId, userText: trimmed, items: [], complete: false }])

      const ws = wsRef.current
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        pendingSendRef.current = trimmed
        connect(sessionId)
        return
      }
      ws.send(JSON.stringify({ type: 'user_message', text: trimmed }))
    },
    [connect, isStreaming, sessionId],
  )

  const respondPermission = useCallback((id: string, answer: 'y' | 'a' | 'n') => {
    wsRef.current?.send(JSON.stringify({ type: 'permission_response', id, answer }))
    const turnId = currentTurnIdRef.current
    if (!turnId) return
    setEntries((prev) =>
      mapTurnItem(prev, turnId, (t) => updateItem(t, id, (i) => (i.kind === 'permission' ? { ...i, resolved: answer } : i))),
    )
  }, [])

  const respondAskUser = useCallback((id: string, answer: string) => {
    wsRef.current?.send(JSON.stringify({ type: 'ask_user_response', id, answer }))
    const turnId = currentTurnIdRef.current
    if (!turnId) return
    setEntries((prev) =>
      mapTurnItem(prev, turnId, (t) => updateItem(t, id, (i) => (i.kind === 'ask_user' ? { ...i, resolved: answer } : i))),
    )
  }, [])

  const startNewChat = useCallback(() => {
    closeSocket()
    currentTurnIdRef.current = null
    pendingTextRef.current = null
    pendingThinkingRef.current = null
    setEntries([])
    setSessionId(null)
    setIsStreaming(false)
    setStatus('idle')
  }, [closeSocket])

  const openSession = useCallback(
    async (id: string) => {
      closeSocket()
      currentTurnIdRef.current = null
      setIsStreaming(false)
      const history = await getSession(id)
      setEntries(history.map((m) => ({ kind: 'message', id: nextId(), role: m.role, content: m.content })))
      setSessionId(id)
      connect(id)
    },
    [closeSocket, connect],
  )

  useEffect(() => closeSocket, [closeSocket])

  return useMemo(
    () => ({ entries, sessionId, status, isStreaming, error, sendMessage, respondPermission, respondAskUser, startNewChat, openSession }),
    [entries, sessionId, status, isStreaming, error, sendMessage, respondPermission, respondAskUser, startNewChat, openSession],
  )
}
