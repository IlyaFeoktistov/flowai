import { useRef, type ChangeEvent } from 'react'
import { IconPaperclip } from '@/shared/ui'
import './AttachButton.css'

export function AttachButton({ onSelect }: { onSelect: (files: File[]) => void }) {
  const inputRef = useRef<HTMLInputElement>(null)

  const onChange = (e: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? [])
    e.target.value = ''
    if (files.length) onSelect(files)
  }

  return (
    <>
      <input ref={inputRef} type="file" multiple className="attach-input" onChange={onChange} />
      <button
        type="button"
        className="attach-btn"
        onClick={() => inputRef.current?.click()}
        aria-label="Прикрепить файл"
        title="Прикрепить файл"
      >
        <IconPaperclip />
      </button>
    </>
  )
}
