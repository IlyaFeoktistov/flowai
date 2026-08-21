import { del, request } from '@/shared/api'

export interface MemorySnapshot {
  facts: string[]
  knowledge: { category: string; key: string; value: string }[]
}

export const getMemory = () => request<MemorySnapshot>('/memory')

export const clearAllMemory = () => del<{ facts: number; knowledge: number }>('/memory')

export const deleteFact = (index: number) => del<{ deleted: boolean }>(`/memory/facts/${index}`)

export const deleteKnowledge = (category: string, key: string) =>
  del<{ deleted: boolean }>('/memory/knowledge', { category, key })
