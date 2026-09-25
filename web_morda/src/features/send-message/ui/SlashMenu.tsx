import { useEffect, useRef } from 'react'
import type { SlashCommand } from '../model/types'

export function SlashMenu({
  items,
  selected,
  onHover,
  onPick,
}: {
  items: SlashCommand[]
  selected: number
  onHover: (index: number) => void
  onPick: (cmd: SlashCommand) => void
}) {
  const listRef = useRef<HTMLUListElement>(null)

  // Стрелками можно уйти за видимую часть списка — докручиваем к выбранному.
  useEffect(() => {
    listRef.current?.children[selected]?.scrollIntoView({ block: 'nearest' })
  }, [selected])

  return (
    <ul className="slash-menu" role="listbox" ref={listRef}>
      {items.map((cmd, i) => (
        <li
          key={cmd.name}
          role="option"
          aria-selected={i === selected}
          className={i === selected ? 'selected' : ''}
          onMouseEnter={() => onHover(i)}
          // mousedown, не click — иначе textarea теряет фокус раньше, чем
          // команда подставится.
          onMouseDown={(e) => {
            e.preventDefault()
            onPick(cmd)
          }}
        >
          <span className="slash-menu-name">{cmd.name}</span>
          <span className="slash-menu-desc">{cmd.description}</span>
        </li>
      ))}
    </ul>
  )
}
