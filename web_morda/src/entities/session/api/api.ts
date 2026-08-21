import { request } from '@/shared/api'
import type { SessionMessage, SessionSummary } from '../model/types'

export const listSessions = () => request<SessionSummary[]>('/sessions')

export const getSession = (id: string) => request<SessionMessage[]>(`/sessions/${encodeURIComponent(id)}`)
