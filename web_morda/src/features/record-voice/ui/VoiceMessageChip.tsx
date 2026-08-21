import { useEffect, useRef, useState, type PointerEvent } from 'react'
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
  const [progress, setProgress] = useState(0) // 0..1, доля волны "проигранной" части
  const [everPlayed, setEverPlayed] = useState(false)
  const [url, setUrl] = useState<string | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const waveRef = useRef<HTMLDivElement>(null)

  // Не useState(() => URL.createObjectURL(...)) — ленивый инициализатор
  // useState тоже двойно вызывается React 18 StrictMode в dev (тот же
  // класс проблемы, что и с nextId() внутри апдейтера setEntries, см.
  // entities/chat's докстринг о живом баге) — createObjectURL создаёт
  // РЕАЛЬНЫЙ браузерный ресурс с побочным эффектом, и без парного cleanup
  // на каждый лишний вызов один из двух созданных URL молча утекал бы
  // (никогда не revoke'store), а какой из двух реально остаётся в сторе —
  // не гарантировано. useEffect с cleanup — правильная пара для ресурса:
  // при двойном вызове (mount→cleanup→mount) первый URL корректно
  // revoke'ится своим же cleanup'ом ДО того, как второй станет активным.
  useEffect(() => {
    const u = URL.createObjectURL(recording.blob)
    setUrl(u)
    return () => URL.revokeObjectURL(u)
  }, [recording.blob])

  // Общий для togglePlay и перемотки по волне: элемент лениво создаётся
  // один раз на url и переиспользуется — пересоздавать его на каждый клик
  // по волне сбрасывало бы уже идущее воспроизведение.
  const ensureAudio = (): HTMLAudioElement | null => {
    if (!url) return null
    if (!audioRef.current || audioRef.current.src !== url) {
      const audio = new Audio(url)
      audio.onended = () => {
        setPlaying(false)
        setProgress(0)
      }
      audio.ontimeupdate = () => {
        if (audio.duration) setProgress(audio.currentTime / audio.duration)
      }
      audioRef.current = audio
    }
    return audioRef.current
  }

  const togglePlay = () => {
    const audio = ensureAudio()
    if (!audio) return
    if (playing) {
      audio.pause()
      setPlaying(false)
    } else {
      setEverPlayed(true)
      audio.play().then(
        () => setPlaying(true),
        () => setPlaying(false),
      )
    }
  }

  // Клик/протяжка по волне — перемотка ТОЛЬКО прослушивания в композере;
  // при отправке уходит весь blob целиком независимо от того, докуда
  // долистали превью (see attachmentToBlock/submit в InputBar — на wire
  // идёт recording.blob как есть, currentTime сюда не просачивается).
  const seekToClientX = (clientX: number) => {
    const el = waveRef.current
    const audio = ensureAudio()
    if (!el || !audio || !Number.isFinite(audio.duration) || audio.duration <= 0) return
    const rect = el.getBoundingClientRect()
    const frac = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width))
    audio.currentTime = frac * audio.duration
    setProgress(frac)
    setEverPlayed(true)
  }

  const onWavePointerDown = (e: PointerEvent<HTMLDivElement>) => {
    seekToClientX(e.clientX)
    e.currentTarget.setPointerCapture(e.pointerId)
  }
  const onWavePointerMove = (e: PointerEvent<HTMLDivElement>) => {
    if (e.buttons !== 1) return
    seekToClientX(e.clientX)
  }

  return (
    <div className="voice-chip">
      <div className="voice-chip-row">
        <button type="button" className="voice-chip-play" onClick={togglePlay} aria-label={playing ? 'Пауза' : 'Прослушать'}>
          {playing ? <IconPause /> : <IconPlay />}
        </button>
        <div
          ref={waveRef}
          className="voice-chip-wave"
          role="slider"
          aria-label="Перемотка прослушивания"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(progress * 100)}
          onPointerDown={onWavePointerDown}
          onPointerMove={onWavePointerMove}
        >
          {recording.waveform.map((v, i) => {
            const played = everPlayed && i / recording.waveform.length <= progress
            return (
              <span
                key={i}
                className={everPlayed ? (played ? 'voice-chip-wave-played' : 'voice-chip-wave-unplayed') : undefined}
                style={{ height: `${6 + v * 18}px` }}
              />
            )
          })}
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
