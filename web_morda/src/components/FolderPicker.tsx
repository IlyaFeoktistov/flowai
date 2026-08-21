import { useEffect, useState } from 'react'
import { api } from '../api'
import { Modal } from './Modal'
import { IconFolder } from './Icons'

export function FolderPicker({
  startPath,
  onClose,
  onChosen,
}: {
  startPath: string
  onClose: () => void
  onChosen: (path: string) => void
}) {
  const [path, setPath] = useState(startPath)
  const [parent, setParent] = useState<string | null>(null)
  const [dirs, setDirs] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = (p?: string) => {
    setError(null)
    api
      .browse(p)
      .then((r) => {
        setPath(r.path)
        setParent(r.parent === r.path ? null : r.parent)
        setDirs(r.dirs)
      })
      .catch((e) => setError(String(e)))
  }

  useEffect(() => {
    load(startPath)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const choose = () => {
    setBusy(true)
    api
      .setProject(path)
      .then(() => onChosen(path))
      .catch((e) => setError(String(e)))
      .finally(() => setBusy(false))
  }

  return (
    <Modal
      title="Выбрать папку"
      onClose={onClose}
      footer={
        <button className="btn btn-primary" onClick={choose} disabled={busy}>
          Открыть эту папку
        </button>
      }
    >
      <div className="folder-path">{path}</div>
      {error && <p className="error-text">{error}</p>}
      <ul className="folder-list">
        {parent !== null && (
          <li className="folder-row" onClick={() => load(parent)}>
            <IconFolder /> ..
          </li>
        )}
        {dirs.map((d) => (
          <li key={d} className="folder-row" onClick={() => load(`${path}/${d}`)}>
            <IconFolder /> {d}
          </li>
        ))}
        {dirs.length === 0 && parent === null && <li className="dim">Нет подпапок</li>}
      </ul>
    </Modal>
  )
}
