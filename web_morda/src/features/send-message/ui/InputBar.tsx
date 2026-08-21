import { useRef, type KeyboardEvent } from 'react'
import { IconSend } from '@/shared/ui'
import { AttachButton } from '@/features/attach-file'
import { MicButton } from '@/features/record-voice'
import './InputBar.css'

export function InputBar({
  streaming,
  queuedCount,
  onSend,
}: {
  streaming: boolean
  queuedCount: number
  onSend: (text: string) => void
}) {
  const ref = useRef<HTMLTextAreaElement>(null)

  const autoGrow = () => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`
  }

  const submit = () => {
    const el = ref.current
    if (!el) return
    const text = el.value
    if (!text.trim()) return
    onSend(text)
    el.value = ''
    el.style.height = 'auto'
  }

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  // Общий вставщик и для прикреплённого файла (уже отформатированный блок
  // "--- имя ---\n...\n---"), и для распознанного голоса (сырой текст) —
  // разница только в том, что кладёт вызывающая сторона.
  const appendText = (text: string) => {
    const el = ref.current
    if (!el) return
    const needsSpace = el.value.length > 0 && !/[\s\n]$/.test(el.value)
    el.value = el.value + (needsSpace ? ' ' : '') + text
    autoGrow()
    el.focus()
  }

  return (
    <div className="input-bar">
      {queuedCount > 0 && <div className="queue-hint">В очереди: {queuedCount}</div>}
      <div className="input-bar-inner">
        <AttachButton onAttach={appendText} />
        <textarea
          ref={ref}
          className="input-textarea"
          placeholder={streaming ? 'Можно продолжать печатать — уйдёт следом за текущим ответом…' : 'Спроси flowAI…'}
          rows={1}
          onKeyDown={onKeyDown}
          onInput={autoGrow}
        />
        <MicButton onTranscribed={appendText} />
        <button className="send-btn" onClick={submit} aria-label="Отправить">
          <IconSend />
        </button>
      </div>
    </div>
  )
}
