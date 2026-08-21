import { useRef, type ChangeEvent } from 'react'
import { IconPaperclip } from '@/shared/ui'
import './AttachButton.css'

// Inlines the file's content directly into the message, same format as
// ui/at_mentions.py's @path resolution (see its docstring) — "\n--- path ---\n
// <content>\n---" — so the model sees it exactly the same shape as the CLI's
// @-mention, whichever entrypoint answers.
export function AttachButton({ onAttach }: { onAttach: (block: string) => void }) {
  const inputRef = useRef<HTMLInputElement>(null)

  const onChange = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (!file) return
    const content = await file.text()
    onAttach(`\n--- ${file.name} ---\n${content}\n---`)
  }

  return (
    <>
      <input ref={inputRef} type="file" className="attach-input" onChange={onChange} />
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
