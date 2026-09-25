import { useEffect, useState } from 'react'
import { getModelParams, setModelParam, type ModelParams } from '../api/api'

// Параметры генерации отдельно для каждой модели (model_params.py) — у
// каждой модели свой набор, по умолчанию открыта текущая chat_model.
export function ModelParamsSection({ chatModel, models }: { chatModel: string; models: string[] }) {
  const [model, setModel] = useState(chatModel)
  const [data, setData] = useState<ModelParams | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    getModelParams(model).then(setData).catch((e) => setError(String(e)))
  }, [model])

  const save = (key: string, raw: string | null) => {
    setError('')
    setModelParam(model, key, raw)
      .then(setData)
      .catch((e) => setError(String(e)))
  }

  const options = models.includes(model) ? models : [model, ...models]

  return (
    <section className="model-params">
      <h3>Параметры модели</h3>
      <select className="settings-input model-params-select" value={model} onChange={(e) => {
          setData(null)
          setModel(e.target.value)
        }}>
        {options.map((m) => (
          <option key={m} value={m}>
            {m}
          </option>
        ))}
      </select>
      {error && <p className="model-params-error">{error}</p>}
      {!data ? (
        <p className="dim">Загружаю…</p>
      ) : (
        <ul className="settings-list">
          {data.params.map((p) => (
            <li key={`${data.model}:${p.key}`} title={p.hint}>
              <span className="settings-key">
                {p.label}
                {p.overridden && <span className="model-params-badge">своё</span>}
              </span>
              <span className="model-params-controls">
                {p.type === 'bool' ? (
                  <input
                    type="checkbox"
                    checked={Boolean(p.value)}
                    onChange={(e) => save(p.key, e.target.checked ? 'true' : 'false')}
                  />
                ) : (
                <input
                  className="settings-input"
                  type="number"
                  step={p.type === 'float' ? 'any' : 1}
                  placeholder={p.default === null ? 'по умолч. бэкенда' : String(p.default)}
                  defaultValue={p.overridden && p.value !== null ? String(p.value) : ''}
                  onBlur={(e) => {
                    const raw = e.target.value.trim()
                    const current = p.overridden && p.value !== null ? String(p.value) : ''
                    if (raw !== current) save(p.key, raw === '' ? null : raw)
                  }}
                />
                )}
                {p.overridden && (
                  <button className="icon-btn" onClick={() => save(p.key, null)} title="Сбросить к значению по умолчанию">
                    сбросить
                  </button>
                )}
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
