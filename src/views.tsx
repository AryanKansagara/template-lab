import { Fragment, type ReactNode } from 'react'
import { segment, type Issue } from './lib/template.ts'

/** Make spaces, tabs and newlines visible while keeping layout. */
export function ws(text: string, show: boolean, key = ''): ReactNode[] {
  if (!show) return [text]
  return text.split(/( +|\t|\n)/).map((part, i) => {
    if (!part) return null
    if (part === '\n')
      return (
        <Fragment key={key + i}>
          <span className="ws nl">↵</span>
          {'\n'}
        </Fragment>
      )
    if (part === '\t') return <span key={key + i} className="ws">{'→\t'}</span>
    if (part[0] === ' ') return <span key={key + i} className="ws">{'·'.repeat(part.length) + '\u200b'}</span>
    return part
  })
}

/** Token text on one line: whitespace becomes symbols, line breaks are the caller's job. */
export function chipText(text: string): ReactNode[] {
  return text.split(/(\s+)/).map((part, i) =>
    /^\s/.test(part) ? (
      <span key={i} className="ws">
        {part.replace(/ /g, '·').replace(/\n/g, '↵').replace(/\t/g, '→').replace(/\r/g, '␍')}
      </span>
    ) : (
      part
    ),
  )
}

/** Same as Rendered's inner part; also used inside diff highlights. */
export function highlight(text: string, specials: string[], showWs: boolean) {
  return segment(text, specials).map((s, i) =>
    s.special ? (
      <span key={i} className="sp">
        {s.text}
      </span>
    ) : (
      <Fragment key={i}>{ws(s.text, showWs, i + ':')}</Fragment>
    ),
  )
}

export function Rendered({ text, specials, showWs }: { text: string; specials: string[]; showWs: boolean }) {
  return (
    <pre className="rendered">
      {highlight(text, specials, showWs)}
    </pre>
  )
}

const MAX_CHIPS = 3000

export function TokenChips({ ids, pieces, specials, showIds }: { ids: number[]; pieces: string[]; specials: Set<string>; showIds: boolean }) {
  return (
    <div className="chips">
      {ids.slice(0, MAX_CHIPS).map((id, i) => {
        const p = pieces[i]
        const sp = specials.has(p)
        return (
          <Fragment key={i}>
            <span className={`chip ${sp ? 'chip-sp' : `c${i % 5}`}`} title={`#${i} · id ${id} · ${JSON.stringify(p)}`}>
              {showIds ? id : p === '' ? '∅' : chipText(p)}
            </span>
            {!showIds && p.endsWith('\n') && <span className="chip-br" />}
          </Fragment>
        )
      })}
      {ids.length > MAX_CHIPS && <span className="muted small">+{(ids.length - MAX_CHIPS).toLocaleString()} more tokens</span>}
    </div>
  )
}

const ICON: Record<Issue['severity'], string> = { error: '✕', warn: '!', info: 'i', ok: '✓' }

export function IssueList({ issues, compact }: { issues: Issue[]; compact?: boolean }) {
  if (!issues.length) return <p className="muted small">Nothing to report.</p>
  return (
    <ul className={`issues ${compact ? 'compact' : ''}`}>
      {issues.map((i) => (
        <li key={i.id} className={`issue sev-${i.severity}`}>
          <span className="issue-icon" aria-label={i.severity}>
            {ICON[i.severity]}
          </span>
          <div>
            <strong>{i.title}</strong>
            {!compact && <p>{i.detail}</p>}
          </div>
        </li>
      ))}
    </ul>
  )
}

export function summarize(issues: Issue[]) {
  const n = (s: Issue['severity']) => issues.filter((i) => i.severity === s).length
  return { error: n('error'), warn: n('warn'), info: n('info'), ok: n('ok') }
}
