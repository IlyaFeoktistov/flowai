import { IconMic } from '@/shared/ui'
import type { RecorderState } from '../model/useVoiceRecorder'
import './MicButton.css'

export function MicButton({ state, onClick }: { state: RecorderState; onClick: () => void }) {
  const label = state === 'recording' ? 'Остановить запись' : 'Голосовой ввод'
  return (
    <button
      type="button"
      className={'mic-btn' + (state === 'recording' ? ' recording' : '')}
      onClick={onClick}
      aria-label={label}
      title={label}
    >
      <IconMic />
    </button>
  )
}
