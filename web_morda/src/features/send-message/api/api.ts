import { request } from '@/shared/api'
import type { SlashCommand } from '../model/types'

// Скилы/плагины текущего проекта — список зависит от os.getcwd() бэкенда,
// поэтому запрашивается заново при каждом открытии меню, а не кешируется.
export const listCommands = () =>
  request<{ commands: SlashCommand[] }>('/commands').then((r) => r.commands)
