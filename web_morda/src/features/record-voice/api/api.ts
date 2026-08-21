import { requestForm } from '@/shared/api'

export const transcribe = (blob: Blob) => {
  const form = new FormData()
  form.append('audio', blob, 'voice.webm')
  return requestForm<{ text: string }>('/transcribe', form)
}
