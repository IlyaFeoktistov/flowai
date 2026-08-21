import { IconClose, IconFileCode, IconFileGeneric } from '@/shared/ui'
import { isCodeFile, type Attachment } from '../model/attachment'
import './AttachmentChip.css'

export function AttachmentChip({ attachment, onRemove }: { attachment: Attachment; onRemove: () => void }) {
  const Icon = attachment.kind === 'image' ? null : isCodeFile(attachment.file) ? IconFileCode : IconFileGeneric

  return (
    <div className="attachment-chip">
      {attachment.kind === 'image' && attachment.previewUrl ? (
        <img className="attachment-chip-thumb" src={attachment.previewUrl} alt="" />
      ) : (
        Icon && <Icon />
      )}
      <span className="attachment-chip-name" title={attachment.file.name}>
        {attachment.file.name}
      </span>
      <button type="button" className="attachment-chip-remove" onClick={onRemove} aria-label="Убрать вложение">
        <IconClose />
      </button>
    </div>
  )
}
