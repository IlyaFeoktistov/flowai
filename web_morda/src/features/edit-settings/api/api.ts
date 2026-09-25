import { postJson, request } from '@/shared/api'

export const getSettings = () => request<Record<string, unknown>>('/settings')

export const setSetting = (key: string, value: unknown) => postJson<{ ok: boolean }>('/settings', { key, value })

export const getModels = () => request<{ models: string[] }>('/models')

export interface ModelParam {
  key: string
  label: string
  type: 'int' | 'float'
  hint: string
  value: number | null
  default: number | null
  overridden: boolean
}

export interface ModelParams {
  model: string
  params: ModelParam[]
}

export const getModelParams = (model?: string) =>
  request<ModelParams>('/model_params' + (model ? `?model=${encodeURIComponent(model)}` : ''))

// value null — сбросить к дефолту модели
export const setModelParam = (model: string, key: string, value: string | null) =>
  postJson<ModelParams>('/model_params', { model, key, value })
