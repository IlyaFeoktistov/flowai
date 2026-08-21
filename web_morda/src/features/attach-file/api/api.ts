import { requestForm } from '@/shared/api'

export const uploadImage = (file: File) => {
  const form = new FormData()
  form.append('image', file)
  return requestForm<{ placeholder: string }>('/upload_image', form)
}
