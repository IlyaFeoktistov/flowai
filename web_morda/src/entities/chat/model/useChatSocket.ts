import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { getSession } from '@/entities/session'
import type { AskUserOption, ConnectionStatus, ConversationEntry, Turn, TurnItem } from './types'

let idSeed = 0
const nextId = () => `${Date.now().toString(36)}-${idSeed++}`

// delegate_tool.py names sub-calls "delegate → <real tool name>" — nest
// those under the delegate card that spawned them instead of showing them
// as flat siblings (see docs/web-ui.md).
const DELEGATE_CHILD_RE = /^delegate → (.+)$/

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

function pushDelegateChild(turn: Turn, id: string, name: string, args: unknown): Turn {
  for (let i = turn.items.length - 1; i >= 0; i--) {
    const item = turn.items[i]
    if (item.kind === 'tool' && item.name === 'delegate' && item.status === 'running') {
      const items = [...turn.items]
      items[i] = { ...item, children: [...(item.children ?? []), { id, name, args, status: 'running' }] }
      return { ...turn, items }
    }
  }
  // Не нашли открытый delegate-родитель (не должно происходить) — не
  // теряем событие молча, показываем плоско.
  return pushItem(turn, { kind: 'tool', id, name: `delegate → ${name}`, args, status: 'running' })
}

function updateDelegateChild(turn: Turn, id: string, result: string): Turn {
  for (let i = turn.items.length - 1; i >= 0; i--) {
    const item = turn.items[i]
    if (item.kind === 'tool' && item.children?.some((c) => c.id === id)) {
      const items = [...turn.items]
      items[i] = {
        ...item,
        children: item.children!.map((c) => (c.id === id ? { ...c, status: 'done', result } : c)),
      }
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
  // Список, не одна строка — ошибки рисуются тостами (см. App.tsx's
  // ToastStack), которые должны стекаться, а не перезаписывать друг друга;
  // id нужен, чтобы ДВЕ ошибки с одинаковым текстом всё равно завели два
  // отдельных тоста, а не молча схлопнулись в один re-render.
  const [errors, setErrors] = useState<{ id: string; message: string }[]>([])
  const pushError = useCallback((message: string) => {
    setErrors((prev) => [...prev, { id: nextId(), message }])
  }, [])
  const dismissError = useCallback((id: string) => {
    setErrors((prev) => prev.filter((e) => e.id !== id))
  }, [])
  // Сколько сообщений уже улетело по сокету, но сервер ещё не подтвердил
  // ни turn_started (стало новым ходом), ни mid_turn_injected (подложено в
  // текущий) — чисто информационная штука для UI, не влияет на порядок/
  // доставку (это делает сервер, см. main.py's inbound/mid_turn_queue).
  const [pendingCount, setPendingCount] = useState(0)

  const wsRef = useRef<WebSocket | null>(null)
  const currentTurnIdRef = useRef<string | null>(null)
  const pendingTextRef = useRef<string | null>(null)
  const pendingThinkingRef = useRef<string | null>(null)
  const pendingSendRef = useRef<string | null>(null)
  // Отправленный текст обычно и есть то, что показывать в пузыре — КРОМЕ
  // вложенных файлов (features/attach-file), где в сообщение подмешивается
  // некрасивый "--- имя ---\n<содержимое>\n---" блок, а в истории хочется
  // видеть то, что человек реально напечатал. sendMessage кладёт сюда
  // (sentText -> displayText), turn_started/mid_turn_injected забирают.
  const displayTextByRawRef = useRef<Map<string, string>>(new Map())

  const closeSocket = useCallback(() => {
    wsRef.current?.close()
    wsRef.current = null
  }, [])

  const displayFor = (raw: string): string => {
    const mapped = displayTextByRawRef.current.get(raw)
    if (mapped !== undefined) displayTextByRawRef.current.delete(raw)
    return mapped ?? raw
  }

  const connect = useCallback((withSessionId: string | null) => {
    closeSocket()
    setStatus('connecting')
    const ws = new WebSocket(wsUrl(withSessionId))
    wsRef.current = ws

    ws.onopen = () => setStatus('open')
    ws.onclose = () => {
      setStatus('closed')
      setIsStreaming(false)
    }
    ws.onerror = () => pushError('Соединение с агентом прервалось')

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
          ws.send(JSON.stringify({ type: 'user_message', text }))
        }
        return
      }

      if (event.type === 'turn_started') {
        const text = event.text as string
        const turnId = nextId()
        currentTurnIdRef.current = turnId
        setIsStreaming(true)
        setPendingCount((n) => Math.max(0, n - 1))
        setEntries((prev) => [
          ...prev,
          { kind: 'turn', id: turnId, userText: displayFor(text), items: [], complete: false, startedAt: Date.now() },
        ])
        return
      }

      const turnId = currentTurnIdRef.current
      if (!turnId) return

      if (event.type === 'mid_turn_injected') {
        setPendingCount((n) => Math.max(0, n - 1))
        const text = displayFor(event.text as string)
        setEntries((prev) => mapTurnItem(prev, turnId, (t) => pushItem(t, { kind: 'mid_turn', id: nextId(), text })))
        return
      }

      // pushError вынесен из свитча ниже нарочно — тот целиком крутится
      // внутри setEntries(prev => ...), а StrictMode-дублирование этой
      // updater-функции задвоило бы тост (тот же класс проблемы, что и с
      // ws.send() в sendMessage, см. её комментарий).
      if (event.type === 'error') {
        pushError(event.message as string)
      }

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

          case 'tool_start': {
            const name = event.name as string
            const id = event.id as string
            const match = DELEGATE_CHILD_RE.exec(name)
            if (match) {
              return mapTurnItem(prev, turnId, (t) => pushDelegateChild(t, id, match[1], event.args))
            }
            return mapTurnItem(prev, turnId, (t) =>
              pushItem(t, {
                kind: 'tool',
                id,
                name,
                args: event.args,
                status: 'running',
                ...(name === 'delegate' ? { children: [] } : {}),
              }),
            )
          }
          case 'tool_end': {
            const name = event.name as string
            const id = event.id as string
            const match = DELEGATE_CHILD_RE.exec(name)
            if (match) {
              return mapTurnItem(prev, turnId, (t) => updateDelegateChild(t, id, event.result as string))
            }
            return mapTurnItem(prev, turnId, (t) =>
              updateItem(t, id, (i) =>
                i.kind === 'tool'
                  ? { ...i, status: 'done', result: event.result as string, diff: event.diff as string | undefined }
                  : i,
              ),
            )
          }

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
            return mapTurnItem(prev, turnId, (t) =>
              pushItem(t, { kind: 'error', id: nextId(), message: event.message as string }),
            )

          case 'stopped':
            return mapTurnItem(prev, turnId, (t) => pushItem(t, { kind: 'stopped', id: nextId() }))

          case 'stats':
            return mapTurnItem(prev, turnId, (t) => ({
              ...t,
              stats: {
                tokensIn: event.tokens_in as number,
                tokensOut: event.tokens_out as number,
                tokensInContent: event.tokens_in_content as number,
                durationMs: event.duration_ms as number,
                genDurationMs: event.gen_duration_ms as number,
                delegateTokensIn: (event.delegate_tokens_in as number) ?? 0,
                delegateTokensOut: (event.delegate_tokens_out as number) ?? 0,
              },
            }))

          case 'turn_complete':
            currentTurnIdRef.current = null
            setIsStreaming(false)
            return mapTurnItem(prev, turnId, (t) => ({ ...t, complete: true, completedAt: Date.now() }))

          default:
            return prev
        }
      })
    }
  }, [closeSocket, pushError])

  // Инпут никогда не блокируется, и сообщение никогда не ждёт на клиенте —
  // уходит по сокету немедленно. Сервер сам решает, что с ним делать: если
  // ход уже идёт на основном агенте — подложит между шагами графа
  // (mid_turn_injected, см. main.py), если ходов нет — станет новым
  // (turn_started); в обоих случаях подтверждение приходит СОБЫТИЕМ, а не
  // сразу здесь, поэтому Turn создаётся в handleEvent, не тут.
  const sendMessage = useCallback(
    (text: string, displayText?: string) => {
      const trimmed = text.trim()
      if (!trimmed) return
      if (displayText && displayText.trim() !== trimmed) {
        displayTextByRawRef.current.set(trimmed, displayText.trim())
      }
      setPendingCount((n) => n + 1)

      const ws = wsRef.current
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        pendingSendRef.current = trimmed
        connect(sessionId)
        return
      }
      ws.send(JSON.stringify({ type: 'user_message', text: trimmed }))
    },
    [connect, sessionId],
  )

  const stopCurrentTurn = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ type: 'stop' }))
  }, [])

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
    displayTextByRawRef.current.clear()
    setPendingCount(0)
    setEntries([])
    setSessionId(null)
    setIsStreaming(false)
    setStatus('idle')
  }, [closeSocket])

  const openSession = useCallback(
    async (id: string) => {
      closeSocket()
      currentTurnIdRef.current = null
      displayTextByRawRef.current.clear()
      setPendingCount(0)
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
    () => ({
      entries,
      sessionId,
      status,
      isStreaming,
      pendingCount,
      errors,
      dismissError,
      sendMessage,
      stopCurrentTurn,
      respondPermission,
      respondAskUser,
      startNewChat,
      openSession,
    }),
    [
      entries,
      sessionId,
      status,
      isStreaming,
      pendingCount,
      errors,
      dismissError,
      sendMessage,
      stopCurrentTurn,
      respondPermission,
      respondAskUser,
      startNewChat,
      openSession,
    ],
  )
}
