import { useRef, type KeyboardEvent } from 'react'
import { IconSend } from './Icons'

export function InputBar({ disabled, onSend }: { disabled: boolean; onSend: (text: string) => void }) {
  const ref = useRef<HTMLTextAreaElement>(null)

  const submit = () => {
    const el = ref.current
    if (!el) return
    const text = el.value
    if (!text.trim() || disabled) return
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

  const autoGrow = () => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`
  }

  return (
    <div className="input-bar">
      <div className="input-bar-inner">
        <textarea
          ref={ref}
          className="input-textarea"
          placeholder={disabled ? 'Генерирует ответ…' : 'Спроси flowAI…'}
          rows={1}
          disabled={disabled}
          onKeyDown={onKeyDown}
          onInput={autoGrow}
        />
        <button className="send-btn" onClick={submit} disabled={disabled} aria-label="Отправить">
          <IconSend />
        </button>
      </div>
    </div>
  )
}
