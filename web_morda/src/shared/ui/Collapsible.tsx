import { useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import './Collapsible.css'

const DEFAULT_MAX_LINES = 10

// Большие ответы/сообщения по умолчанию обрезаются по высоте (em — чтобы
// совпадать с line-height родителя, а не фиксированный px), кнопка
// "Показать полностью" появляется только если контент реально не влез —
// re-measure на каждое изменение children, чтобы поток токенов (answer_chunk)
// сам себя "доразворачивал" в тумблер, как только пересечёт порог.
export function Collapsible({ children, maxLines = DEFAULT_MAX_LINES }: { children: ReactNode; maxLines?: number }) {
  const bodyRef = useRef<HTMLDivElement>(null)
  const [overflowing, setOverflowing] = useState(false)
  const [expanded, setExpanded] = useState(false)

  useLayoutEffect(() => {
    const el = bodyRef.current
    if (!el) return
    setOverflowing(el.scrollHeight > el.clientHeight + 1)
  }, [children])

  return (
    <div className="collapsible">
      <div
        ref={bodyRef}
        className={'collapsible-body' + (!expanded && overflowing ? ' collapsible-clamped' : '')}
        style={expanded ? undefined : { maxHeight: `${maxLines * 1.6}em` }}
      >
        {children}
      </div>
      {overflowing && (
        <button type="button" className="collapsible-toggle" onClick={() => setExpanded((v) => !v)}>
          {expanded ? 'Свернуть' : 'Показать полностью'}
        </button>
      )}
    </div>
  )
}
