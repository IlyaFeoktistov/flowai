import { IconMic } from '@/shared/ui'
import { useVoiceRecorder } from '../model/useVoiceRecorder'
import './MicButton.css'

export function MicButton({ onTranscribed }: { onTranscribed: (text: string) => void }) {
  const { state, toggle } = useVoiceRecorder(onTranscribed)

  const label =
    state === 'recording' ? 'Остановить запись' : state === 'transcribing' ? 'Распознаю…' : 'Голосовой ввод'

  return (
    <button
      type="button"
      className={'mic-btn' + (state === 'recording' ? ' recording' : '')}
      onClick={toggle}
      disabled={state === 'transcribing'}
      aria-label={label}
      title={label}
    >
      <IconMic />
    </button>
  )
}
