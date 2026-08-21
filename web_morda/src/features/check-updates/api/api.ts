import { postJson } from '@/shared/api'

export const checkUpdates = () => postJson<{ report: string }>('/update')
