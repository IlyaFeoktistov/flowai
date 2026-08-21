import { useEffect, useState } from 'react'
import type { ToolChild, Turn, TurnItem } from '@/entities/chat'
import { renderMarkdown } from '@/shared/lib'
import { Collapsible, IconThinking, IconTool } from '@/shared/ui'

// Те же "прикольные надписи" и тот же ~4 символа/токен эвристик, что
// ui/stream.py's _THINKING_PHRASES/_tok_approx — портированы буквально,
// не придуманы заново, чтобы веб выглядел как терминал, а не как отдельный
// стиль (см. _ai_header/_format_duration там же).
const THINKING_PHRASES = [
  'думаю', 'шевелю извилинами', 'включаю мозги', 'подбираю слова',
  'варю мысли', 'советуюсь с нейронами', 'гружу идеи',
]
const TOOL_RUNNING_PHRASES = [
  'выполняю тулы', 'копаюсь в файлах', 'дёргаю рычаги',
  'колдую с инструментами', 'жму на кнопки', 'кручу гайки',
]
const PROCESSING_PHRASES = [
  'обрабатываю', 'перевариваю результат', 'раскладываю по полочкам',
  'сверяю показания', 'анализирую улов', 'изучаю добычу',
]
const GENERATING_PHRASES = ['печатаю', 'строчу ответ', 'накидываю мысль', 'выдаю мысль', 'пишу']

type Phase = 'thinking' | 'tool' | 'processing' | 'generating'

function currentPhase(items: TurnItem[]): Phase | null {
  const last = items[items.length - 1]
  if (!last) return null
  switch (last.kind) {
    case 'thinking':
      return last.open ? 'thinking' : 'processing'
    case 'tool':
      return last.status === 'running' ? 'tool' : 'processing'
    case 'text':
      return last.open ? 'generating' : null
    case 'stage':
    case 'plan':
    case 'mid_turn':
      return 'processing'
    default:
      return null
  }
}

function phrasesFor(phase: Phase): readonly string[] {
  switch (phase) {
    case 'thinking': return THINKING_PHRASES
    case 'tool': return TOOL_RUNNING_PHRASES
    case 'processing': return PROCESSING_PHRASES
    case 'generating': return GENERATING_PHRASES
  }
}

function estimateTokens(items: TurnItem[]): number {
  let chars = 0
  for (const it of items) {
    if (it.kind === 'thinking' || it.kind === 'text') chars += it.text.length
  }
  return Math.max(0, Math.round(chars / 4))
}

function formatDuration(totalSeconds: number): string {
  const s = Math.max(0, Math.floor(totalSeconds))
  const secs = s % 60
  const mins = Math.floor(s / 60) % 60
  const hours = Math.floor(s / 3600)
  if (hours) return `${hours}ч ${mins}м ${secs}с`
  if (mins) return `${mins}м ${secs}с`
  return `${secs}с`
}

// Живой тикающий счётчик, пока ход не завершён (оценка по символам — тот же
// компромисс, что и в терминале, точных чисел от модели ещё нет), и точные
// цифры из "stats"-события (agent.py/pipeline.py) после. Фраза перевыбирается
// только при смене фазы, не на каждый тик — иначе рябило бы каждые 400мс.
function TurnFooter({ turn }: { turn: Turn }) {
  const [now, setNow] = useState(() => Date.now())
  const [phrase, setPhrase] = useState('')
  const phase = currentPhase(turn.items)

  useEffect(() => {
    if (turn.complete) return
    const id = setInterval(() => setNow(Date.now()), 400)
    return () => clearInterval(id)
  }, [turn.complete])

  // Перевыбирается только когда фаза реально меняется (эффект, а не во
  // время рендера) — иначе StrictMode-двойной вызов рендера или прерванный
  // concurrent-рендер могли бы перекатить фразу без смены фазы.
  useEffect(() => {
    setPhrase(phase ? phrasesFor(phase)[Math.floor(Math.random() * phrasesFor(phase).length)] : '')
  }, [phase])

  const endMs = turn.completedAt ?? now
  const elapsedSec = (endMs - turn.startedAt) / 1000
  const tok = turn.stats ? turn.stats.tokensOut : estimateTokens(turn.items)
  if (tok === 0 && elapsedSec < 1) return null

  const label = !turn.complete && phrase ? `${phrase} · ` : ''
  const delegateTok = turn.stats && (turn.stats.delegateTokensIn || turn.stats.delegateTokensOut)
    ? turn.stats.delegateTokensIn + turn.stats.delegateTokensOut
    : 0

  return (
    <div className="turn-footer">
      {label}
      {tok} tok · {formatDuration(elapsedSec)}
      {delegateTok > 0 && <span className="turn-footer-dim"> (из них делегат: {delegateTok} tok)</span>}
    </div>
  )
}

const STAGE_LABELS: Record<string, string> = {
  analyzer: 'Анализатор',
  planner: 'Планировщик',
  coder: 'Кодер',
  verifier: 'Верификатор',
  casual: 'Ответ',
  quick_fix: 'Быстрая правка',
  main: 'Основной агент',
}

function formatArgs(args: unknown): string {
  if (args == null) return ''
  if (typeof args === 'string') return args
  try {
    return JSON.stringify(args, null, 2)
  } catch {
    return String(args)
  }
}

function ToolBody({ argsLabel, argsText, result }: { argsLabel: string; argsText: string; result?: string }) {
  return (
    <div className="turn-tool-body">
      {argsText && (
        <div>
          <div className="turn-tool-label">{argsLabel}</div>
          <pre className="code-block">{argsText}</pre>
        </div>
      )}
      {result !== undefined && (
        <div>
          <div className="turn-tool-label">Результат</div>
          <pre className="code-block">{result}</pre>
        </div>
      )}
    </div>
  )
}

function ToolChildRow({ child }: { child: ToolChild }) {
  const argsText = formatArgs(child.args)
  return (
    <details className="turn-tool turn-tool-child">
      <summary>
        <span className={'tool-dot' + (child.status === 'running' ? ' running' : '')} />
        <span className="tool-name">{child.name}</span>
        {argsText && <span className="tool-args-preview">{argsText.slice(0, 60).replace(/\s+/g, ' ')}</span>}
      </summary>
      <ToolBody argsLabel="Аргументы" argsText={argsText} result={child.result} />
    </details>
  )
}

// delegate — не обычный тул: сам вызов сворачиваемый (как любой другой), а
// под ним ВСЕГДА видно, какие sub-tool-calls он сделал (delegate_tool.py's
// "delegate → X" события, см. entities/chat's pushDelegateChild) — то, ради
// чего это отдельный компонент, а не ToolItem с доп. пропом.
function DelegateItem({ item }: { item: Extract<TurnItem, { kind: 'tool' }> }) {
  const argsText = formatArgs(item.args)
  return (
    <div className="turn-delegate">
      <details className="turn-tool">
        <summary>
          <span className={'tool-dot' + (item.status === 'running' ? ' running' : '')} />
          <IconTool />
          <span className="tool-name">delegate</span>
          {argsText && <span className="tool-args-preview">{argsText.slice(0, 80).replace(/\s+/g, ' ')}</span>}
        </summary>
        <ToolBody argsLabel="Задача" argsText={argsText} result={item.result} />
      </details>
      {item.children && item.children.length > 0 && (
        <ul className="delegate-children">
          {item.children.map((c) => (
            <li key={c.id}>
              <ToolChildRow child={c} />
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function ToolItem({ item }: { item: Extract<TurnItem, { kind: 'tool' }> }) {
  if (item.name === 'delegate') return <DelegateItem item={item} />
  const argsText = formatArgs(item.args)
  return (
    <details className="turn-tool">
      <summary>
        <span className={'tool-dot' + (item.status === 'running' ? ' running' : '')} />
        <IconTool />
        <span className="tool-name">{item.name}</span>
        {argsText && <span className="tool-args-preview">{argsText.slice(0, 80).replace(/\s+/g, ' ')}</span>}
      </summary>
      <ToolBody argsLabel="Аргументы" argsText={argsText} result={item.result} />
    </details>
  )
}

function PlanItem({ item }: { item: Extract<TurnItem, { kind: 'plan' }> }) {
  return (
    <ol className="turn-plan">
      {item.steps.map((step, i) => (
        <li key={i} className={item.doneIndexes.includes(i) ? 'done' : undefined}>
          {step}
        </li>
      ))}
    </ol>
  )
}

function PermissionItem({
  item,
  onRespond,
}: {
  item: Extract<TurnItem, { kind: 'permission' }>
  onRespond: (id: string, answer: 'y' | 'a' | 'n') => void
}) {
  return (
    <div className="turn-permission">
      <div className="turn-permission-detail">
        <span className="warn-label">Запрос разрешения · {item.action}</span>
        <pre className="code-block">{item.detail}</pre>
      </div>
      {item.resolved ? (
        <div className="dim">
          {{ y: 'Разрешено', a: 'Разрешено всегда', n: 'Отклонено' }[item.resolved]}
        </div>
      ) : (
        <div className="btn-row">
          <button className="btn btn-primary" onClick={() => onRespond(item.id, 'y')}>
            Да
          </button>
          <button className="btn" onClick={() => onRespond(item.id, 'a')}>
            Да, всегда
          </button>
          <button className="btn btn-danger" onClick={() => onRespond(item.id, 'n')}>
            Нет
          </button>
        </div>
      )}
    </div>
  )
}

function AskUserItem({
  item,
  onRespond,
}: {
  item: Extract<TurnItem, { kind: 'ask_user' }>
  onRespond: (id: string, answer: string) => void
}) {
  const [customAnswer, setCustomAnswer] = useState('')

  const submitCustom = () => {
    const text = customAnswer.trim()
    if (text) onRespond(item.id, text)
  }

  return (
    <div className="turn-ask-user">
      <div className="ask-user-question">{item.question}</div>
      {item.resolved ? (
        <div className="dim">Ответ: {item.resolved}</div>
      ) : (
        <>
          {/* Стопкой, не строкой пилюль — так помещается описание под
              лейблом (раньше оно пряталось в title-тултип и было не видно
              вообще), и не ломается на дублирующихся лейблах — модель
              иногда честно присылает два варианта с одним и тем же
              текстом, но разными description (см. index, не opt.label, —
              одинаковый key на дублях уже путал React). */}
          <ul className="ask-user-options">
            {item.options.map((opt, i) => (
              <li key={i}>
                <button
                  className={'ask-user-option' + (opt.label === item.recommended ? ' ask-user-option-recommended' : '')}
                  onClick={() => onRespond(item.id, opt.label)}
                >
                  <span className="ask-user-option-label">{opt.label}</span>
                  {opt.description && <span className="ask-user-option-desc">{opt.description}</span>}
                </button>
              </li>
            ))}
          </ul>
          <div className="ask-user-custom">
            <input
              type="text"
              className="ask-user-custom-input"
              placeholder="Свой вариант ответа…"
              value={customAnswer}
              onChange={(e) => setCustomAnswer(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') submitCustom()
              }}
            />
            <button type="button" className="btn" onClick={submitCustom} disabled={!customAnswer.trim()}>
              Отправить
            </button>
          </div>
        </>
      )}
    </div>
  )
}

function TurnItemView({
  item,
  onRespondPermission,
  onRespondAskUser,
}: {
  item: TurnItem
  onRespondPermission: (id: string, answer: 'y' | 'a' | 'n') => void
  onRespondAskUser: (id: string, answer: string) => void
}) {
  switch (item.kind) {
    case 'stage':
      return <div className="turn-stage">{STAGE_LABELS[item.stage] ?? item.stage}</div>
    case 'thinking':
      return (
        <details className="turn-thinking" open={item.open}>
          <summary>
            <IconThinking /> Размышляет…
          </summary>
          <div className="turn-thinking-body">{item.text}</div>
        </details>
      )
    case 'text':
      // Без Collapsible нарочно — это ответ модели, не вложение/тул-
      // результат: он и так обычно короткий, сворачивать его в 10 строк
      // по умолчанию больше мешало, чем помогало.
      return <div className="turn-text">{renderMarkdown(item.text)}</div>
    case 'tool':
      return <ToolItem item={item} />
    case 'plan':
      return <PlanItem item={item} />
    case 'permission':
      return <PermissionItem item={item} onRespond={onRespondPermission} />
    case 'ask_user':
      return <AskUserItem item={item} onRespond={onRespondAskUser} />
    case 'error':
      return <div className="turn-error">{item.message}</div>
    case 'mid_turn':
      return <div className="turn-mid-injected">📤 Добавлено по ходу: {item.text}</div>
    case 'stopped':
      return <div className="turn-stopped">⏹ Остановлено пользователем</div>
  }
}

export function TurnView({
  turn,
  onRespondPermission,
  onRespondAskUser,
}: {
  turn: Turn
  onRespondPermission: (id: string, answer: 'y' | 'a' | 'n') => void
  onRespondAskUser: (id: string, answer: string) => void
}) {
  return (
    <div className="turn">
      <div className="msg msg-user">
        <div className="msg-bubble">
          <Collapsible>{turn.userText}</Collapsible>
        </div>
      </div>
      <div className="turn-detail">
        {turn.items.map((item) => (
          <TurnItemView
            key={item.id}
            item={item}
            onRespondPermission={onRespondPermission}
            onRespondAskUser={onRespondAskUser}
          />
        ))}
        {!turn.complete && <span className="turn-live-dot" aria-label="генерирует" />}
      </div>
      <TurnFooter turn={turn} />
    </div>
  )
}
