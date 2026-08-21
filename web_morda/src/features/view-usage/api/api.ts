import { request } from '@/shared/api'

export const usage = () => request<Record<string, number>>('/usage')
