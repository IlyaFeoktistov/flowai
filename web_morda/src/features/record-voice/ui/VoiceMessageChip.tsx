import { useEffect, useRef, useState } from 'react'
import { IconClose, IconPause, IconPlay } from '@/shared/ui'
import type { VoiceRecording } from '../model/useVoiceRecorder'
import './VoiceMessageChip.css'

function formatDuration(ms: number): string {
  const s = Math.round(ms / 1000)
  return `0:${String(s).padStart(2, '0')}`
}

// Композер-превью голосового до отправки, "как в мессенджерах" — волна
// реальной амплитуды (useVoiceRecorder), проиграть можно сразу, а
// транскрипция — по кнопке "T", не автоматически (см. record-voice/api's
// transcribe). transcript/transcribing — управляемые пропсы, не локальный
// стейт: InputBar должен знать текст на момент отправки, а не только чип.
export function VoiceMessageChip({
  recording,
  transcript,
  transcribing,
  onTranscribe,
  onRemove,
}: {
  recording: VoiceRecording
  transcript: string | null
  transcribing: boolean
  onTranscribe: () => void
  onRemove: () => void
}) {
  const [playing, setPlaying] = useState(false)
  const [url] = useState(() => URL.createObjectURL(recording.blob))
  const audioRef = useRef<HTMLAudioElement | null>(null)

  useEffect(() => () => URL.revokeObjectURL(url), [url])

  const togglePlay = () => {
    if (!audioRef.current) {
      audioRef.current = new Audio(url)
      audioRef.current.onended = () => setPlaying(false)
    }
    if (playing) {
      audioRef.current.pause()
      setPlaying(false)
    } else {
      audioRef.current.play()
      setPlaying(true)
    }
  }

  return (
    <div className="voice-chip">
      <div className="voice-chip-row">
        <button type="button" className="voice-chip-play" onClick={togglePlay} aria-label={playing ? 'Пауза' : 'Прослушать'}>
          {playing ? <IconPause /> : <IconPlay />}
        </button>
        <div className="voice-chip-wave" aria-hidden="true">
          {recording.waveform.map((v, i) => (
            <span key={i} style={{ height: `${6 + v * 18}px` }} />
          ))}
        </div>
        <span className="voice-chip-duration">{formatDuration(recording.durationMs)}</span>
        <button
          type="button"
          className="voice-chip-transcribe"
          onClick={onTranscribe}
          disabled={transcribing || transcript !== null}
          title="Транскрибировать"
        >
          {transcribing ? '…' : 'T'}
        </button>
        <button type="button" className="voice-chip-remove" onClick={onRemove} aria-label="Убрать голосовое">
          <IconClose />
        </button>
      </div>
      {transcript !== null && <div className="voice-chip-transcript">{transcript}</div>}
    </div>
  )
}
