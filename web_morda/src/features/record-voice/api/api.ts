import { requestForm } from '@/shared/api'

export const transcribe = (blob: Blob) => {
  const form = new FormData()
  form.append('audio', blob, 'voice.webm')
  return requestForm<{ text: string }>('/transcribe', form)
}

// Для войс-режима (VoiceOrb) — отдаёт синтезированный WAV как blob, чтобы
// проиграть его через <audio> в браузере. См. main.py:/speak.
export async function speak(text: string): Promise<Blob> {
  const res = await fetch('/api/v1/speak', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
  if (!res.ok) throw new Error(`${res.status} /speak`)
  return res.blob()
}
