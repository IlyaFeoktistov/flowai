import { uploadImage } from '../api/api'

export interface Attachment {
  id: string
  file: File
  kind: 'image' | 'text'
  previewUrl?: string
}

let idSeed = 0
const nextId = () => `att-${Date.now().toString(36)}-${idSeed++}`

const CODE_EXT_RE = /\.(py|js|jsx|ts|tsx|go|rs|java|kt|c|cpp|h|hpp|cs|rb|php|sh|css|scss|html|json|ya?ml|toml|sql|md)$/i

export function classifyFile(file: File): 'image' | 'text' {
  return file.type.startsWith('image/') ? 'image' : 'text'
}

export function isCodeFile(file: File): boolean {
  return CODE_EXT_RE.test(file.name)
}

export function toAttachment(file: File): Attachment {
  const kind = classifyFile(file)
  return {
    id: nextId(),
    file,
    kind,
    previewUrl: kind === 'image' ? URL.createObjectURL(file) : undefined,
  }
}

export function releaseAttachment(att: Attachment): void {
  if (att.previewUrl) URL.revokeObjectURL(att.previewUrl)
}

// Инлайнится в текст сообщения ПРЯМО ПЕРЕД отправкой, не в момент
// прикрепления — композер до этого держит только чипы (см. InputBar).
// Текстовые файлы — тем же форматом, что ui/at_mentions.py's @путь;
// картинки — через ui/images.py's store_image (main.py:/upload_image) —
// плейсхолдер [Image-N], который process_turns резолвит в реальный путь
// прямо перед тем, как текст увидит модель.
export async function attachmentToBlock(att: Attachment): Promise<string> {
  if (att.kind === 'image') {
    const { placeholder } = await uploadImage(att.file)
    return `\n${placeholder}`
  }
  const content = await att.file.text()
  return `\n--- ${att.file.name} ---\n${content}\n---`
}
