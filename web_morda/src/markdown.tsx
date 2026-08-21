import type { ReactNode } from 'react'

// Small hand-rolled renderer instead of a markdown dependency — the model's
// output is almost always plain prose + fenced code blocks + inline code/
// bold, and that's what this covers. Not a general CommonMark implementation.

let keySeed = 0
const nextKey = () => `md-${keySeed++}`

function renderInline(text: string): ReactNode[] {
  const nodes: ReactNode[] = []
  // inline code | bold | italic, in that precedence order
  const re = /(`[^`]+`)|(\*\*[^*]+\*\*)|(\*[^*]+\*)/g
  let last = 0
  let match: RegExpExecArray | null
  while ((match = re.exec(text))) {
    if (match.index > last) nodes.push(text.slice(last, match.index))
    const token = match[0]
    if (token.startsWith('`')) {
      nodes.push(<code key={nextKey()}>{token.slice(1, -1)}</code>)
    } else if (token.startsWith('**')) {
      nodes.push(<strong key={nextKey()}>{token.slice(2, -2)}</strong>)
    } else {
      nodes.push(<em key={nextKey()}>{token.slice(1, -1)}</em>)
    }
    last = re.lastIndex
  }
  if (last < text.length) nodes.push(text.slice(last))
  return nodes
}

export function renderMarkdown(source: string): ReactNode {
  const parts = source.split(/```(\w*)\n?([\s\S]*?)```/g)
  // String.split with a capturing regex interleaves: [plain, lang, code, plain, lang, code, ...]
  const blocks: ReactNode[] = []
  for (let i = 0; i < parts.length; i += 3) {
    const plain = parts[i]
    if (plain) {
      const paragraphs = plain.split(/\n{2,}/).filter((p) => p.trim())
      for (const para of paragraphs) {
        blocks.push(
          <p key={nextKey()}>
            {para.split('\n').flatMap((line, idx, arr) =>
              idx < arr.length - 1 ? [...renderInline(line), <br key={nextKey()} />] : renderInline(line),
            )}
          </p>,
        )
      }
    }
    const lang = parts[i + 1]
    const code = parts[i + 2]
    if (code !== undefined) {
      blocks.push(
        <pre key={nextKey()} className="code-block">
          {lang && <div className="code-block-lang">{lang}</div>}
          <code>{code.replace(/\n$/, '')}</code>
        </pre>,
      )
    }
  }
  return blocks
}
