import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
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
}: {
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

  const submit = async () => {
    const el = ref.current
    const typed = el?.value.trim() ?? ''
    if (!typed && attachments.length === 0 && !pendingVoice) return

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
  }

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
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
        <AttachButton onSelect={onAttachFiles} />
        <textarea
          ref={ref}
          className="input-textarea"
          placeholder={streaming ? 'Можно писать дальше — сообщение подключится к текущему ответу…' : 'Спроси flowAI…'}
          rows={1}
          onKeyDown={onKeyDown}
          onInput={autoGrow}
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
    </div>
  )
}
