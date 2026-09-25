import { postJson, request } from '@/shared/api'

export interface ModelInstance {
  id: string
  backend: 'ollama' | 'llama.cpp'
  model: string
  ram_bytes: number | null
  vram_bytes: number | null
  vram_estimated: boolean
  details: string
}

export const listInstances = () => request<{ instances: ModelInstance[]; errors: string[] }>('/instances')

export const unloadInstance = (id: string) => postJson<{ message: string }>('/instances/unload', { id })
