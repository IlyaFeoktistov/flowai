import { useCallback, useRef, useState } from 'react'
import { transcribe } from '../api/api'

export type RecorderState = 'idle' | 'recording' | 'transcribing'

// Toggle-to-record, not a fixed duration — click starts, click again (or
// calling toggle() a second time) stops and uploads for transcription.
// Recording happens entirely in the browser (MediaRecorder); the backend
// only does STT (ui/audio.py's transcribe(), faster-whisper) on the
// resulting blob — see main.py's /transcribe.
export function useVoiceRecorder(onTranscribed: (text: string) => void) {
  const [state, setState] = useState<RecorderState>('idle')
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const streamRef = useRef<MediaStream | null>(null)

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      chunksRef.current = []
      const recorder = new MediaRecorder(stream)
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }
      recorder.onstop = async () => {
        streamRef.current?.getTracks().forEach((t) => t.stop())
        streamRef.current = null
        setState('transcribing')
        try {
          const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' })
          const { text } = await transcribe(blob)
          if (text) onTranscribed(text)
        } catch {
          // мик/сеть подвели — тихо возвращаемся в idle, текста просто не будет
        } finally {
          setState('idle')
        }
      }
      recorderRef.current = recorder
      recorder.start()
      setState('recording')
    } catch {
      setState('idle')
    }
  }, [onTranscribed])

  const toggle = useCallback(() => {
    if (state === 'recording') {
      recorderRef.current?.stop()
    } else if (state === 'idle') {
      start()
    }
  }, [state, start])

  return { state, toggle }
}
