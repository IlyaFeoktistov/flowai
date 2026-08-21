import type { ToolChild, Turn, TurnItem } from '@/entities/chat'
import { renderMarkdown } from '@/shared/lib'
import { IconThinking, IconTool } from '@/shared/ui'

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
  return (
    <div className="turn-ask-user">
      <div className="ask-user-question">{item.question}</div>
      {item.resolved ? (
        <div className="dim">Ответ: {item.resolved}</div>
      ) : (
        <div className="btn-row wrap">
          {item.options.map((opt) => (
            <button
              key={opt.label}
              className={'btn' + (opt.label === item.recommended ? ' btn-primary' : '')}
              title={opt.description}
              onClick={() => onRespond(item.id, opt.label)}
            >
              {opt.label}
            </button>
          ))}
        </div>
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
        <div className="msg-bubble">{turn.userText}</div>
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
    </div>
  )
}
