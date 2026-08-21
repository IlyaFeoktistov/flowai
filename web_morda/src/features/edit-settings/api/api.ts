import { postJson, request } from '@/shared/api'

export const getSettings = () => request<Record<string, unknown>>('/settings')

export const setSetting = (key: string, value: unknown) => postJson<{ ok: boolean }>('/settings', { key, value })
