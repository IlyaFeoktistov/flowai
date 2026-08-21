const BASE = '/api/v1'

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
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

export const postJson = <T,>(path: string, body?: unknown) =>
  request<T>(path, { method: 'POST', body: body === undefined ? undefined : JSON.stringify(body) })

export const del = <T,>(path: string, body?: unknown) =>
  request<T>(path, { method: 'DELETE', body: body === undefined ? undefined : JSON.stringify(body) })
