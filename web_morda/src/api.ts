import type { SessionSummary, StoredMessage } from './types'

const BASE = '/api/v1'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    let detail = ''
    try {
      detail = JSON.stringify(await res.json())
    } catch {
      detail = await res.text()
    }
    throw new Error(`${res.status} ${path}: ${detail}`)
  }
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

const postJson = <T,>(path: string, body?: unknown) =>
  request<T>(path, { method: 'POST', body: body === undefined ? undefined : JSON.stringify(body) })

const del = <T,>(path: string, body?: unknown) =>
  request<T>(path, { method: 'DELETE', body: body === undefined ? undefined : JSON.stringify(body) })

export const api = {
  getProject: () => request<{ path: string }>('/project'),
  setProject: (path: string) => postJson<{ path: string }>('/project', { path }),
  browse: (path?: string) =>
    request<{ path: string; parent: string; dirs: string[] }>(
      `/browse${path ? `?path=${encodeURIComponent(path)}` : ''}`,
    ),

  listSessions: () => request<SessionSummary[]>('/sessions'),
  getSession: (id: string) => request<StoredMessage[]>(`/sessions/${encodeURIComponent(id)}`),

  doctor: () => request<{ report: string }>('/doctor'),
  update: () => postJson<{ report: string }>('/update'),
  cleanReport: () => request<{ report: string }>('/clean'),
  cleanApply: (scope: string) => postJson<{ report: string }>('/clean', { scope }),
  usage: () => request<Record<string, number>>('/usage'),

  memory: () =>
    request<{ facts: string[]; knowledge: { category: string; key: string; value: string }[] }>('/memory'),
  memoryClearAll: () => del<{ facts: number; knowledge: number }>('/memory'),
  memoryDeleteFact: (index: number) => del<{ deleted: boolean }>(`/memory/facts/${index}`),
  memoryDeleteKnowledge: (category: string, key: string) =>
    del<{ deleted: boolean }>('/memory/knowledge', { category, key }),

  plugins: () => request<{ report: string }>('/plugins'),
  reindex: (targets?: string[]) => postJson<Record<string, unknown>>('/reindex', { targets: targets ?? null }),

  settingsGet: () => request<Record<string, unknown>>('/settings'),
  settingsSet: (key: string, value: unknown) => postJson<{ ok: boolean }>('/settings', { key, value }),
}
