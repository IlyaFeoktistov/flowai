import type { ReactNode } from 'react'
import katex from 'katex'

// Small hand-rolled renderer instead of a markdown dependency — the model's
// output is almost always plain prose + fenced code blocks + inline code/
// bold + the occasional table/formula, and that's what this covers. Not a
// general CommonMark implementation.

let keySeed = 0
const nextKey = () => `md-${keySeed++}`

function renderMath(tex: string, displayMode: boolean): ReactNode {
  try {
    const html = katex.renderToString(tex.trim(), { displayMode, throwOnError: false })
    // katex's own output — not user-controlled HTML, this is the standard
    // way to mount it (katex has no React renderer of its own).
    return <span key={nextKey()} dangerouslySetInnerHTML={{ __html: html }} />
  } catch {
    return <code key={nextKey()}>{tex}</code>
  }
}

function renderInline(text: string): ReactNode[] {
  const nodes: ReactNode[] = []
  // inline math \( \) | inline code | bold | italic, in that precedence order
  const re = /(\\\([\s\S]+?\\\))|(`[^`]+`)|(\*\*[^*]+\*\*)|(\*[^*]+\*)/g
  let last = 0
  let match: RegExpExecArray | null
  while ((match = re.exec(text))) {
    if (match.index > last) nodes.push(text.slice(last, match.index))
    const token = match[0]
    if (token.startsWith('\\(')) {
      nodes.push(renderMath(token.slice(2, -2), false))
    } else if (token.startsWith('`')) {
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

function splitRow(line: string): string[] {
  let trimmed = line.trim()
  if (trimmed.startsWith('|')) trimmed = trimmed.slice(1)
  if (trimmed.endsWith('|')) trimmed = trimmed.slice(0, -1)
  return trimmed.split('|').map((c) => c.trim())
}

function looksLikeSeparatorRow(line: string): boolean {
  if (!line.includes('-')) return false
  const cells = splitRow(line)
  return cells.length > 0 && cells.every((c) => /^:?-{1,}:?$/.test(c))
}

function renderTable(header: string[], rows: string[][]): ReactNode {
  return (
    <div className="table-scroll" key={nextKey()}>
      <table>
        <thead>
          <tr>
            {header.map((h, i) => (
              <th key={i}>{renderInline(h)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, ri) => (
            <tr key={ri}>
              {r.map((c, ci) => (
                <td key={ci}>{renderInline(c)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// A block of plain text (already stripped of fenced code / display-math
// blocks by renderMarkdown below) — still needs paragraph vs. GFM-table
// detection, since a table is just consecutive `|`-lines with a
// `|---|---|`-style separator as its second line, not its own fenced syntax.
function renderPlainBlock(plain: string): ReactNode[] {
  const blocks: ReactNode[] = []
  const paragraphs = plain.split(/\n{2,}/).filter((p) => p.trim())
  for (const para of paragraphs) {
    const lines = para.split('\n')
    let textBuf: string[] = []
    const flushText = () => {
      if (textBuf.length === 0) return
      blocks.push(
        <p key={nextKey()}>
          {textBuf.flatMap((line, idx, arr) =>
            idx < arr.length - 1 ? [...renderInline(line), <br key={nextKey()} />] : renderInline(line),
          )}
        </p>,
      )
      textBuf = []
    }
    let i = 0
    while (i < lines.length) {
      const isHeaderRow = lines[i].includes('|') && i + 1 < lines.length && looksLikeSeparatorRow(lines[i + 1])
      if (isHeaderRow) {
        flushText()
        const header = splitRow(lines[i])
        const rows: string[][] = []
        let j = i + 2
        while (j < lines.length && lines[j].includes('|')) {
          rows.push(splitRow(lines[j]))
          j++
        }
        blocks.push(renderTable(header, rows))
        i = j
      } else {
        textBuf.push(lines[i])
        i++
      }
    }
    flushText()
  }
  return blocks
}

// Top-level blocks that must NOT be treated as prose: fenced code, and
// display math ($$...$$ or \[...\] — the model is told to use the latter,
// see prompts.py's math_notation_rule, but both are supported since models
// mix conventions).
const BLOCK_RE = /```(\w*)\n?([\s\S]*?)```|\$\$([\s\S]*?)\$\$|\\\[([\s\S]*?)\\\]/g

export function renderMarkdown(source: string): ReactNode {
  const blocks: ReactNode[] = []
  let last = 0
  let match: RegExpExecArray | null
  BLOCK_RE.lastIndex = 0
  while ((match = BLOCK_RE.exec(source))) {
    if (match.index > last) blocks.push(...renderPlainBlock(source.slice(last, match.index)))
    const [, lang, code, dollarMath, bracketMath] = match
    if (code !== undefined) {
      blocks.push(
        <pre key={nextKey()} className="code-block">
          {lang && <div className="code-block-lang">{lang}</div>}
          <code>{code.replace(/\n$/, '')}</code>
        </pre>,
      )
    } else {
      blocks.push(
        <div key={nextKey()} className="math-block">
          {renderMath((dollarMath ?? bracketMath) as string, true)}
        </div>,
      )
    }
    last = BLOCK_RE.lastIndex
  }
  if (last < source.length) blocks.push(...renderPlainBlock(source.slice(last)))
  return blocks
}
