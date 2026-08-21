import { request } from '@/shared/api'

export const doctor = () => request<{ report: string }>('/doctor')
