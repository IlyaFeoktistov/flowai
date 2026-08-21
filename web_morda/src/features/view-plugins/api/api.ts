import { request } from '@/shared/api'

export const plugins = () => request<{ report: string }>('/plugins')
