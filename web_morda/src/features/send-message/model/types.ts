export interface SlashCommand {
  name: string
  description: string
  // Есть — команда выполняется прямо во фронтенде (модалка, новый чат) и
  // на бэкенд не уходит; нет — текст отправляется как обычное сообщение,
  // бэкенд сам разберёт /plan, /build и скилы.
  run?: (args: string) => void
}
