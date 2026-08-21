import { useCallback, useRef, useState } from 'react'

export type RecorderState = 'idle' | 'recording'

export interface VoiceRecording {
  blob: Blob
  waveform: number[] // normalized 0..1 amplitude bars, fixed length (see BAR_COUNT)
  durationMs: number
}

const BAR_COUNT = 32

function downsampleToBars(samples: number[], count: number): number[] {
  if (samples.length === 0) return new Array(count).fill(0.1)
  const bars: number[] = []
  const bucket = samples.length / count
  for (let i = 0; i < count; i++) {
    const from = Math.floor(i * bucket)
    const to = Math.max(from + 1, Math.floor((i + 1) * bucket))
    let peak = 0
    for (let j = from; j < to && j < samples.length; j++) peak = Math.max(peak, samples[j])
    bars.push(Math.max(0.08, peak))
  }
  return bars
}

// Toggle-to-record, not a fixed duration — click starts, click again stops.
// Recording (MediaRecorder) AND the waveform (AnalyserNode RMS sampled via
// requestAnimationFrame) both happen client-side; onRecorded gets the raw
// blob + a real amplitude waveform (not a fake/static one) once stopped —
// transcription is a SEPARATE, on-demand step (see record-voice/api's
// transcribe + the "T" button on VoiceMessageChip), not automatic here.
export function useVoiceRecorder(onRecorded: (recording: VoiceRecording) => void) {
  const [state, setState] = useState<RecorderState>('idle')
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const streamRef = useRef<MediaStream | null>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const samplesRef = useRef<number[]>([])
  const rafRef = useRef(0)
  const startedAtRef = useRef(0)

  // Named function expression (not a useCallback referencing its own const
  // binding) — a plain recursive rAF loop, sidesteps the "read while its
  // declaration is being initialized" lint warning a self-referencing
  // useCallback would trigger, even though that pattern is safe here too.
  const sampleLoopRef = useRef<() => void>(undefined)
  sampleLoopRef.current = function sampleLoop() {
    const analyser = analyserRef.current
    if (!analyser) return
    const data = new Uint8Array(analyser.fftSize)
    analyser.getByteTimeDomainData(data)
    let sumSq = 0
    for (let i = 0; i < data.length; i++) {
      const v = (data[i] - 128) / 128
      sumSq += v * v
    }
    const rms = Math.sqrt(sumSq / data.length)
    samplesRef.current.push(Math.min(1, rms * 4))
    rafRef.current = requestAnimationFrame(() => sampleLoopRef.current?.())
  }

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      chunksRef.current = []
      samplesRef.current = []
      startedAtRef.current = Date.now()

      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
      const audioCtx = new AudioCtx()
      const source = audioCtx.createMediaStreamSource(stream)
      const analyser = audioCtx.createAnalyser()
      analyser.fftSize = 512
      source.connect(analyser)
      audioCtxRef.current = audioCtx
      analyserRef.current = analyser
      rafRef.current = requestAnimationFrame(() => sampleLoopRef.current?.())

      const recorder = new MediaRecorder(stream)
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }
      recorder.onstop = () => {
        cancelAnimationFrame(rafRef.current)
        streamRef.current?.getTracks().forEach((t) => t.stop())
        streamRef.current = null
        audioCtxRef.current?.close().catch(() => {})
        audioCtxRef.current = null
        analyserRef.current = null

        // Клик "стоп" почти сразу после клика "старт" (или отпущенный/
        // забракованный микрофон посреди записи) — ondataavailable мог не
        // успеть выстрелить вообще ни разу, тогда chunksRef.current пуст, а
        // Blob([], {...}) — 0 байт, гарантированно неиграбельный (та самая
        // "no supported source"). Не отдаём такую запись наверх молча.
        if (chunksRef.current.length === 0) {
          setState('idle')
          return
        }

        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' })
        const durationMs = Date.now() - startedAtRef.current
        onRecorded({ blob, waveform: downsampleToBars(samplesRef.current, BAR_COUNT), durationMs })
        setState('idle')
      }
      recorderRef.current = recorder
      recorder.start()
      setState('recording')
    } catch {
      setState('idle')
    }
  }, [onRecorded])

  const toggle = useCallback(() => {
    if (state === 'recording') {
      recorderRef.current?.stop()
    } else {
      start()
    }
  }, [state, start])

  return { state, toggle }
}
