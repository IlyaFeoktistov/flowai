import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { IconSend, IconStop } from '@/shared/ui'
import { AttachButton, AttachmentChip, attachmentToBlock, releaseAttachment, toAttachment, type Attachment } from '@/features/attach-file'
import {
  MicButton,
  VoiceMessageChip,
  VoiceOrb,
  speak,
  transcribe,
  useVoiceRecorder,
  type VoiceRecording,
} from '@/features/record-voice'
import { listCommands } from '../api/api'
import type { SlashCommand } from '../model/types'
import { SlashMenu } from './SlashMenu'
import './InputBar.css'

interface PendingVoice {
  recording: VoiceRecording
  transcript: string | null
  transcribing: boolean
}

export function InputBar({
  streaming,
  pendingCount,
  lastAnswerText,
  onSend,
  onStop,
  context,
  workMode,
  onWorkModeChange,
  builtinCommands,
}: {
  builtinCommands: SlashCommand[]
  context: { tokens: number; limit: number | null } | null
  workMode: 'plan' | 'build'
  onWorkModeChange: (mode: 'plan' | 'build') => void
  streaming: boolean
  pendingCount: number
  lastAnswerText: string | null
  onSend: (text: string, displayText?: string) => void
  onStop: () => void
}) {
  const ref = useRef<HTMLTextAreaElement>(null)
  const [attachments, setAttachments] = useState<Attachment[]>([])
  const [pendingVoice, setPendingVoice] = useState<PendingVoice | null>(null)
  // Была ли ПОСЛЕДНЯЯ отправка голосовой — решает, нужно ли озвучивать
  // ответ и показывать шарик, пока ждём его (см. эффект ниже).
  const [awaitingVoiceReply, setAwaitingVoiceReply] = useState(false)
  const [speaking, setSpeaking] = useState(false)
  const wasStreamingRef = useRef(streaming)

  // Слеш-меню — как _CmdCompleter в ui/app.py: видно, пока ввод начинается
  // с "/" и в нём ещё нет пробела (дальше идут аргументы команды).
  const [draft, setDraft] = useState('')
  const [skillCommands, setSkillCommands] = useState<SlashCommand[]>([])
  const [menuIndex, setMenuIndex] = useState(0)
  const [menuDismissed, setMenuDismissed] = useState(false)
  const menuQuery = draft.startsWith('/') && !/\s/.test(draft) ? draft : null
  const menuOpen = menuQuery !== null

  useEffect(() => {
    if (!menuOpen) return
    listCommands().then(setSkillCommands).catch(() => setSkillCommands([]))
  }, [menuOpen])

  const menuItems = useMemo(() => {
    if (menuQuery === null) return []
    // Встроенные не перекрываются скилом с тем же именем — как в CLI.
    const builtinNames = new Set(builtinCommands.map((c) => c.name))
    return [...builtinCommands, ...skillCommands.filter((c) => !builtinNames.has(c.name))].filter((c) =>
      c.name.startsWith(menuQuery),
    )
  }, [menuQuery, builtinCommands, skillCommands])
  const showMenu = !menuDismissed && menuItems.length > 0

  const { state: recorderState, toggle: toggleRecording } = useVoiceRecorder((recording) => {
    setPendingVoice({ recording, transcript: null, transcribing: false })
  })

  // Ход, начатый голосом, закончился — озвучиваем ответ (TTS), шарик
  // продолжает "бурлить" всё это время (см. VoiceOrb).
  useEffect(() => {
    if (wasStreamingRef.current && !streaming && awaitingVoiceReply) {
      setAwaitingVoiceReply(false)
      const text = (lastAnswerText ?? '').trim()
      if (text) {
        setSpeaking(true)
        speak(text)
          .then((blob) => {
            const url = URL.createObjectURL(blob)
            const audio = new Audio(url)
            audio.onended = () => {
              setSpeaking(false)
              URL.revokeObjectURL(url)
            }
            audio.onerror = () => {
              setSpeaking(false)
              URL.revokeObjectURL(url)
            }
            audio.play().catch(() => setSpeaking(false))
          })
          .catch(() => setSpeaking(false))
      }
    }
    wasStreamingRef.current = streaming
  }, [streaming, awaitingVoiceReply, lastAnswerText])

  const autoGrow = () => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`
  }

  const onAttachFiles = (files: File[]) => {
    setAttachments((prev) => [...prev, ...files.map(toAttachment)])
  }

  const removeAttachment = (id: string) => {
    setAttachments((prev) => {
      const found = prev.find((a) => a.id === id)
      if (found) releaseAttachment(found)
      return prev.filter((a) => a.id !== id)
    })
  }

  const transcribeVoice = async () => {
    if (!pendingVoice || pendingVoice.transcript !== null) return
    setPendingVoice((p) => (p ? { ...p, transcribing: true } : p))
    try {
      const { text } = await transcribe(pendingVoice.recording.blob)
      setPendingVoice((p) => (p ? { ...p, transcript: text, transcribing: false } : p))
    } catch {
      setPendingVoice((p) => (p ? { ...p, transcribing: false } : p))
    }
  }

  const setText = (text: string) => {
    const el = ref.current
    if (!el) return
    el.value = text
    el.setSelectionRange(text.length, text.length)
    setDraft(text)
    setMenuIndex(0)
    setMenuDismissed(false)
    autoGrow()
  }

  // Локальные команды (модалки, новый чат) выполняются во фронтенде и в
  // чат не уходят; всё остальное — /plan, /build, скилы — обычное
  // сообщение, бэкенд разбирает его сам.
  const runLocal = (text: string): boolean => {
    const [head, ...rest] = text.split(' ')
    const cmd = builtinCommands.find((c) => c.name === head)
    if (!cmd?.run) return false
    cmd.run(rest.join(' ').trim())
    setText('')
    return true
  }

  const pickCommand = (cmd: SlashCommand) => {
    if (cmd.run) {
      runLocal(cmd.name)
      return
    }
    setText(cmd.name + ' ')
    ref.current?.focus()
  }

  const submit = async () => {
    const el = ref.current
    const typed = el?.value.trim() ?? ''
    if (!typed && attachments.length === 0 && !pendingVoice) return
    if (typed.startsWith('/') && attachments.length === 0 && !pendingVoice && runLocal(typed)) return

    let voiceText = ''
    const usedVoice = !!pendingVoice
    if (pendingVoice) {
      voiceText = pendingVoice.transcript ?? ''
      if (voiceText === '') {
        try {
          const { text } = await transcribe(pendingVoice.recording.blob)
          voiceText = text
        } catch {
          voiceText = ''
        }
      }
    }

    const blocks = await Promise.all(attachments.map(attachmentToBlock))
    const displayText = [typed, voiceText].filter(Boolean).join('\n') || typed
    const fullText = [typed, voiceText].filter(Boolean).join('\n') + blocks.join('')
    if (!fullText.trim() && blocks.length === 0) return

    onSend(fullText, displayText)
    if (usedVoice) setAwaitingVoiceReply(true)

    attachments.forEach(releaseAttachment)
    setAttachments([])
    setPendingVoice(null)
    if (el) {
      el.value = ''
      el.style.height = 'auto'
    }
    setDraft('')
  }

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (showMenu) {
      const selected = menuItems[Math.min(menuIndex, menuItems.length - 1)]
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault()
        const step = e.key === 'ArrowDown' ? 1 : -1
        setMenuIndex((i) => (i + step + menuItems.length) % menuItems.length)
        return
      }
      if (e.key === 'Escape') {
        e.preventDefault()
        setMenuDismissed(true)
        return
      }
      if (e.key === 'Tab') {
        e.preventDefault()
        setText(selected.name + ' ')
        return
      }
      // Enter по уже полностью набранной команде — отправка (как в CLI);
      // по недописанной — сначала подставить выбранную.
      if (e.key === 'Enter' && !e.shiftKey && selected.name !== draft) {
        e.preventDefault()
        pickCommand(selected)
        return
      }
    }
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  const showOrb = recorderState === 'recording' || (streaming && awaitingVoiceReply) || speaking
  const orbLabel = recorderState === 'recording' ? 'Говорю…' : speaking ? 'Отвечает…' : 'Жду ответ…'

  return (
    <div className="input-bar">
      {pendingCount > 0 && <div className="queue-hint">В обработке: {pendingCount}</div>}
      {showOrb && <VoiceOrb label={orbLabel} />}
      {(attachments.length > 0 || pendingVoice) && (
        <div className="composer-attachments">
          {attachments.map((a) => (
            <AttachmentChip key={a.id} attachment={a} onRemove={() => removeAttachment(a.id)} />
          ))}
          {pendingVoice && (
            <VoiceMessageChip
              recording={pendingVoice.recording}
              transcript={pendingVoice.transcript}
              transcribing={pendingVoice.transcribing}
              onTranscribe={transcribeVoice}
              onRemove={() => setPendingVoice(null)}
            />
          )}
        </div>
      )}
      <div className="input-bar-inner">
        {showMenu && (
          <SlashMenu
            items={menuItems}
            selected={Math.min(menuIndex, menuItems.length - 1)}
            onHover={setMenuIndex}
            onPick={pickCommand}
          />
        )}
        <AttachButton onSelect={onAttachFiles} />
        <textarea
          ref={ref}
          className="input-textarea"
          placeholder={streaming ? 'Можно писать дальше — сообщение подключится к текущему ответу…' : 'Спроси FlowAI…'}
          rows={1}
          onKeyDown={onKeyDown}
          onInput={(e) => {
            setDraft(e.currentTarget.value)
            setMenuIndex(0)
            setMenuDismissed(false)
            autoGrow()
          }}
        />
        <MicButton state={recorderState} onClick={toggleRecording} />
        {streaming ? (
          <button className="send-btn stop-btn" onClick={onStop} aria-label="Остановить">
            <IconStop />
          </button>
        ) : (
          <button className="send-btn" onClick={submit} aria-label="Отправить">
            <IconSend />
          </button>
        )}
      </div>
      <div className="input-bar-meta">
        <div className="work-mode-toggle" role="group" aria-label="Режим работы">
          <button
            className={workMode === 'plan' ? 'active plan' : ''}
            onClick={() => onWorkModeChange('plan')}
            title="Исследование только на чтение, правки заблокированы. План — по просьбе, сохранится в .flowai/plans/"
          >
            План
          </button>
          <button
            className={workMode === 'build' ? 'active' : ''}
            onClick={() => onWorkModeChange('build')}
            title="Агент правит код"
          >
            Build
          </button>
        </div>
      {context && (
        <div className={'context-usage' + (context.limit && context.tokens / context.limit >= 0.8 ? ' high' : '')}>
          контекст {(context.tokens / 1000).toFixed(1)}k
          {context.limit ? ` / ${Math.round(context.limit / 1000)}k (${Math.floor((context.tokens * 100) / context.limit)}%)` : ''}
        </div>
      )}
      </div>
    </div>
  )
}
