import { useEffect } from 'react'
import { IconClose } from './Icons'
import './ToastStack.css'

const AUTO_DISMISS_MS = 8000

export interface Toast {
  id: string
  message: string
}

// Каждый тост несёт свой собственный таймер — не общий на весь стек,
// иначе досрочный клик по одному сбросил бы обратный отсчёт остальных.
function ToastItem({ toast, onDismiss }: { toast: Toast; onDismiss: (id: string) => void }) {
  useEffect(() => {
    const t = setTimeout(() => onDismiss(toast.id), AUTO_DISMISS_MS)
    return () => clearTimeout(t)
  }, [toast.id, onDismiss])

  return (
    <div className="toast" role="alert">
      <span className="toast-message">{toast.message}</span>
      <button type="button" className="toast-close" onClick={() => onDismiss(toast.id)} aria-label="Закрыть">
        <IconClose />
      </button>
    </div>
  )
}

// Справа сверху, стопкой — новые добавляются снизу, ничего друг друга не
// перекрывает (flex-column + gap, не absolute-позиционирование).
export function ToastStack({ toasts, onDismiss }: { toasts: Toast[]; onDismiss: (id: string) => void }) {
  if (toasts.length === 0) return null
  return (
    <div className="toast-stack">
      {toasts.map((t) => (
        <ToastItem key={t.id} toast={t} onDismiss={onDismiss} />
      ))}
    </div>
  )
}
