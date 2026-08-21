import { postJson, request } from '@/shared/api'

export const cleanReport = () => request<{ report: string }>('/clean')

export const cleanApply = (scope: string) => postJson<{ report: string }>('/clean', { scope })
