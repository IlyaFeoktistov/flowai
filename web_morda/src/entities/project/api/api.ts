import { postJson, request } from '@/shared/api'
import type { BrowseResult, ProjectInfo } from '../model/types'

export const getProject = () => request<ProjectInfo>('/project')

export const setProject = (path: string) => postJson<ProjectInfo>('/project', { path })

export const browse = (path?: string) =>
  request<BrowseResult>(`/browse${path ? `?path=${encodeURIComponent(path)}` : ''}`)
