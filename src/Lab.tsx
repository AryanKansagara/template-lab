import { useDeferredValue, useEffect, useMemo, useState } from 'react'
import { DEFAULT_MODELS, fetchModelTemplate, HubError, type ModelTemplate } from './lib/hub.ts'
import { CHATML_FALLBACK, diff, diffUnits, lint, render, templateKwargs, type ChatMessage, type Issue, type RenderResult, type Role, type SpecialTokens } from './lib/template.ts'
import { decodeState, encodeState } from './lib/share.ts'
import { SAMPLES, WEATHER_TOOLS, type UIMsg } from './samples.ts'
import { tokenizerAddsBos, useTokenizer, type TokState } from './tokenizer.ts'
import { highlight, IssueList, Rendered, summarize, TokenChips } from './views.tsx'

interface LabState {
  /** Hub model id; '' in paste mode */
  model: string
  /** user-edited / pasted template; null = use the Hub one */
  override: string | null
  tplName: string
  pasteBos: string
  pasteEos: string
  compare: string
  msgs: UIMsg[]
  toolsOn: boolean
  tools: string
  agp: boolean
  kwargs: string
  tokenize: boolean
}

const DEFAULT: LabState = {
  model: 'Qwen/Qwen2.5-7B-Instruct',
  override: null,
  tplName: 'default',
  pasteBos: '',
  pasteEos: '',
  compare: 'mistralai/Mistral-7B-Instruct-v0.3',
  msgs: SAMPLES[0].msgs,
  toolsOn: false,
  tools: WEATHER_TOOLS,
  agp: true,
  kwargs: '{}',
  tokenize: false,
}

/** Shared links and localStorage are untrusted: keep only well-shaped fields. */
function sanitize(raw: unknown): Partial<LabState> {
  if (!raw || typeof raw !== 'object') return {}
  const out: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(raw)) {
    if (k in DEFAULT && typeof v === typeof DEFAULT[k as keyof LabState] && k !== 'msgs' && k !== 'override') out[k] = v
  }
  const r = raw as { override?: unknown; msgs?: unknown }
  if (typeof r.override === 'string') out.override = r.override
  if (Array.isArray(r.msgs)) {
    out.msgs = r.msgs
      .filter((m) => m && ['system', 'user', 'assistant', 'tool'].includes(m.role) && typeof m.content === 'string')
      .map((m) => ({ role: m.role, content: m.content, ...(typeof m.toolCalls === 'string' ? { toolCalls: m.toolCalls } : {}), ...(typeof m.toolCallId === 'string' ? { toolCallId: m.toolCallId } : {}) }))
  }
  return out
}

const STORE = 'tl:state'
const PASTE_TOKENIZER = 'Qwen/Qwen2.5-7B-Instruct'

// ---------------------------------------------------------------- hub loading

type HubState = { status: 'idle' } | { status: 'loading' } | { status: 'ready'; model: ModelTemplate } | { status: 'error'; error: HubError }
const hubCache = new Map<string, Promise<ModelTemplate>>()

function useHubModel(id: string): HubState {
  const [s, setS] = useState<HubState>({ status: 'idle' })
  useEffect(() => {
    if (!id) return setS({ status: 'idle' })
    let live = true
    let p = hubCache.get(id)
    if (!p) {
      p = fetchModelTemplate(id)
      p.catch(() => hubCache.delete(id))
      hubCache.set(id, p)
    }
    setS({ status: 'loading' })
    p.then(
      (model) => live && setS({ status: 'ready', model }),
      (e) => live && setS({ status: 'error', error: e instanceof HubError ? e : new HubError('network', String(e?.message ?? e)) }),
    )
    return () => {
      live = false
    }
  }, [id])
  return s
}

// ---------------------------------------------------------------- input conversion

function parseJson(label: string, text: string, errors: string[]): unknown {
  if (!text.trim()) return undefined
  try {
    return JSON.parse(text)
  } catch (e) {
    errors.push(`${label}: ${(e as Error).message}`)
    return undefined
  }
}

function toChat(msgs: UIMsg[], errors: string[]): ChatMessage[] {
  return msgs.map((m, i) => {
    const out: ChatMessage = { role: m.role, content: m.content }
    if (m.role === 'assistant' && m.toolCalls?.trim()) {
      const calls = parseJson(`Message #${i + 1} tool_calls`, m.toolCalls, errors)
      if (Array.isArray(calls)) out.tool_calls = calls
      else if (calls !== undefined) errors.push(`Message #${i + 1} tool_calls must be a JSON array`)
    }
    if (m.role === 'tool' && m.toolCallId) out.tool_call_id = m.toolCallId
    return out
  })
}

interface Built {
  template: string | null
  specials: SpecialTokens
  specialList: string[]
  result: RenderResult
  issues: Issue[]
}

function build(template: string | null, specials: SpecialTokens, specialList: string[], base: Parameters<typeof lint>[0]['input'], addsBos: boolean | null): Built {
  const input = { ...base, specials }
  return {
    template,
    specials,
    specialList,
    result: render({ ...input, template: template?.trim() ? template : CHATML_FALLBACK }),
    issues: lint({ template, input, tokenizerAddsBos: addsBos }),
  }
}

function tokenize(t: TokState, text: string) {
  if (t.status !== 'ready') return null
  const ids = t.tok.encode(text, { add_special_tokens: false })
  return { ids, pieces: ids.map((id) => t.tok.decode([id], { skip_special_tokens: false })) }
}

// ---------------------------------------------------------------- component

export default function Lab() {
  const [state, setState] = useState<LabState>(() => {
    try {
      return { ...DEFAULT, ...sanitize(JSON.parse(localStorage.getItem(STORE) ?? '{}')) }
    } catch {
      return DEFAULT
    }
  })
  const set = (patch: Partial<LabState>) => setState((s) => ({ ...s, ...patch }))
  const [tab, setTab] = useState<'rendered' | 'tokens' | 'compare' | 'template'>('rendered')
  const [showWs, setShowWs] = useState(true)
  const [showIds, setShowIds] = useState(false)
  const [mode, setMode] = useState<'hub' | 'paste'>(() => (state.model ? 'hub' : 'paste'))
  const [draft, setDraft] = useState(state.model)
  const [compareDraft, setCompareDraft] = useState(state.compare)
  const [toast, setToast] = useState('')

  // Shared link wins over saved state.
  useEffect(() => {
    const applyHash = () => {
    const preset = location.hash.match(/^#try=([\w-]+)\|(.+)$/)
    if (preset) {
      const sample = SAMPLES.find((x) => x.id === preset[1])
      const model = decodeURIComponent(preset[2])
      if (sample) {
        set({ msgs: sample.msgs, agp: sample.agp, toolsOn: !!sample.tools, tools: sample.tools ?? WEATHER_TOOLS, kwargs: sample.kwargs ?? '{}' })
        loadModel(model)
      }
      history.replaceState(null, '', location.pathname)
      return
    }
    const m = location.hash.match(/^#s=(.+)$/)
    if (!m) return
    decodeState<Partial<LabState>>(m[1]).then((s) => {
      if (!s) return flash('That share link is broken or truncated.')
      const next = { ...DEFAULT, ...sanitize(s) }
      setState(next)
      setDraft(next.model)
      setCompareDraft(next.compare)
      setMode(next.model ? 'hub' : 'paste')
      history.replaceState(null, '', location.pathname)
    })
    }
    applyHash()
    addEventListener('hashchange', applyHash)
    return () => removeEventListener('hashchange', applyHash)
  }, [])

  useEffect(() => localStorage.setItem(STORE, JSON.stringify(state)), [state])

  function flash(msg: string) {
    setToast(msg)
    setTimeout(() => setToast(''), 2600)
  }

  const hubA = useHubModel(state.model)
  const hubB = useHubModel(tab === 'compare' ? state.compare : '')
  const modelA = hubA.status === 'ready' ? hubA.model : null
  const modelB = hubB.status === 'ready' ? hubB.model : null

  const tokIdA = state.model || PASTE_TOKENIZER
  const tokA = useTokenizer(tokIdA, state.tokenize)
  const tokB = useTokenizer(modelB ? state.compare : null, state.tokenize && tab === 'compare')

  const deferred = useDeferredValue(state)
  const errors: string[] = []
  const input = useMemo(() => {
    const errs: string[] = []
    const kwargs = parseJson('Extra kwargs', deferred.kwargs, errs)
    const tools = deferred.toolsOn ? parseJson('Tools', deferred.tools, errs) : undefined
    if (tools !== undefined && !Array.isArray(tools)) errs.push('Tools must be a JSON array')
    if (kwargs !== undefined && (typeof kwargs !== 'object' || Array.isArray(kwargs) || kwargs === null)) errs.push('Extra kwargs must be a JSON object')
    return {
      errs,
      messages: toChat(deferred.msgs, errs),
      tools: Array.isArray(tools) ? tools : null,
      kwargs: kwargs && typeof kwargs === 'object' && !Array.isArray(kwargs) ? (kwargs as Record<string, unknown>) : {},
      addGenerationPrompt: deferred.agp,
    }
  }, [deferred.kwargs, deferred.tools, deferred.toolsOn, deferred.msgs, deferred.agp])
  errors.push(...input.errs)

  const hubTemplate = modelA ? (modelA.templates.find((t) => t.name === deferred.tplName) ?? modelA.templates[0])?.template ?? null : null
  const pasteSpecials: SpecialTokens = { bos_token: deferred.pasteBos || undefined, eos_token: deferred.pasteEos || undefined }

  const A = useMemo<Built | null>(() => {
    if (mode === 'hub' && !modelA) return null
    const template = deferred.override ?? hubTemplate
    const specials = modelA && mode === 'hub' ? modelA.specials : pasteSpecials
    const list = modelA && mode === 'hub' ? modelA.specialTokens : Object.values(pasteSpecials).filter(Boolean) as string[]
    const adds = tokA.status === 'ready' ? tokenizerAddsBos(tokA.tok, specials.bos_token) : mode === 'hub' ? (modelA?.addBosToken ?? null) : null
    return build(mode === 'paste' ? deferred.override ?? '' : template, specials, list, input, adds)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, modelA, hubTemplate, deferred.override, deferred.pasteBos, deferred.pasteEos, input, tokA])

  const B = useMemo<Built | null>(() => {
    if (!modelB) return null
    const adds = tokB.status === 'ready' ? tokenizerAddsBos(tokB.tok, modelB.specials.bos_token) : modelB.addBosToken
    return build(modelB.templates[0]?.template ?? null, modelB.specials, modelB.specialTokens, input, adds)
  }, [modelB, input, tokB])

  const tokensA = useMemo(() => (A?.result.ok ? tokenize(tokA, A.result.text) : null), [A, tokA])
  const tokensB = useMemo(() => (B?.result.ok ? tokenize(tokB, B.result.text) : null), [B, tokB])

  const counts = A ? summarize(A.issues) : null
  const kwargsDetected = useMemo(() => templateKwargs(A?.template ?? ''), [A?.template])

  async function share() {
    const s = await encodeState(state)
    const url = `${location.origin}/app#s=${s}`
    try {
      await navigator.clipboard.writeText(url)
      flash('Share link copied to clipboard')
    } catch {
      prompt('Copy this link', url)
    }
  }

  function loadModel(id: string) {
    const clean = id.trim()
    if (!clean) return
    setDraft(clean)
    setMode('hub')
    set({ model: clean, override: null, tplName: 'default' })
  }

  function loadSample(id: string) {
    const s = SAMPLES.find((x) => x.id === id)
    if (!s) return
    set({ msgs: s.msgs, agp: s.agp, toolsOn: !!s.tools, tools: s.tools ?? state.tools, kwargs: s.kwargs ?? '{}' })
    if (s.model && mode === 'hub') loadModel(s.model)
    flash(`Loaded sample: ${s.label}`)
  }

  const updateMsg = (i: number, patch: Partial<UIMsg>) => set({ msgs: state.msgs.map((m, j) => (j === i ? { ...m, ...patch } : m)) })
  const moveMsg = (i: number, d: number) => {
    const msgs = [...state.msgs]
    const [m] = msgs.splice(i, 1)
    msgs.splice(i + d, 0, m)
    set({ msgs })
  }
  const addMsg = (role: Role) => set({ msgs: [...state.msgs, { role, content: '', ...(role === 'tool' ? { toolCallId: '' } : {}) }] })

  function addKwarg(k: string) {
    let obj: Record<string, unknown> = {}
    try {
      obj = JSON.parse(state.kwargs || '{}')
    } catch {
      /* replace broken JSON */
    }
    if (!(k in obj)) obj[k] = k.startsWith('enable_') || k.endsWith('_mode') ? false : k === 'date_string' ? new Date().toDateString().slice(4) : ''
    set({ kwargs: JSON.stringify(obj, null, 2) })
  }

  return (
    <div className="lab">
      {/* ------------------------------------------------ source bar */}
      <section className="card source" aria-label="Template source">
        <div className="source-top">
          <div className="seg" role="tablist" aria-label="Template source">
            <button role="tab" aria-selected={mode === 'hub'} onClick={() => { setMode('hub'); if (!state.model) loadModel(draft || DEFAULT.model) }}>
              From the Hub
            </button>
            <button role="tab" aria-selected={mode === 'paste'} onClick={() => { setMode('paste'); set({ model: '', override: state.override ?? A?.template ?? '', pasteBos: modelA?.specials.bos_token ?? state.pasteBos, pasteEos: modelA?.specials.eos_token ?? state.pasteEos }); setTab('template') }}>
              Paste a template
            </button>
          </div>
          <div className="source-actions">
            <label className="select-wrap">
              <span className="sr-only">Load a sample conversation</span>
              <select value="" onChange={(e) => loadSample(e.target.value)}>
                <option value="">Load sample…</option>
                {SAMPLES.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.label}
                  </option>
                ))}
              </select>
            </label>
            <button className="btn" onClick={share} title="Copy a link with this exact setup">
              <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M10 13a5 5 0 0 0 7.5.5l3-3a5 5 0 0 0-7-7l-1.7 1.7" /><path d="M14 11a5 5 0 0 0-7.5-.5l-3 3a5 5 0 0 0 7 7l1.7-1.7" /></svg>
              Share
            </button>
          </div>
        </div>

        {mode === 'hub' ? (
          <>
            <form className="model-form" onSubmit={(e) => { e.preventDefault(); loadModel(draft) }}>
              <label className="sr-only" htmlFor="model-id">Model id</label>
              <div className="input-icon">
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></svg>
                <input id="model-id" className="input mono" value={draft} onChange={(e) => setDraft(e.target.value)} placeholder="org/model, e.g. Qwen/Qwen2.5-7B-Instruct, or paste a huggingface.co URL" spellCheck={false} autoComplete="off" />
              </div>
              <button className="btn btn-dark" type="submit" disabled={hubA.status === 'loading' && draft === state.model}>
                {hubA.status === 'loading' && draft === state.model ? 'Loading…' : 'Load'}
              </button>
            </form>
            <div className="quick" aria-label="Popular models">
              {DEFAULT_MODELS.map((id) => (
                <button key={id} className={`pill ${state.model === id ? 'on' : ''}`} onClick={() => loadModel(id)}>
                  {id.split('/')[1]}
                </button>
              ))}
              <button className="pill ghost" onClick={() => loadModel('meta-llama/Llama-3.1-8B-Instruct')} title="Gated repo: see how it's handled">
                Llama-3.1 (gated)
              </button>
            </div>
            <ModelMeta hub={hubA} onPick={loadModel} override={state.override !== null} onReset={() => set({ override: null })} tplName={state.tplName} onTpl={(n) => set({ tplName: n })} />
          </>
        ) : (
          <div className="paste-meta">
            <p className="muted small">Edit the template in the <button className="linkish" onClick={() => setTab('template')}>Template</button> tab. Special tokens the template reads:</p>
            <label>
              <span>bos_token</span>
              <input className="input mono sm" value={state.pasteBos} onChange={(e) => set({ pasteBos: e.target.value })} placeholder="<s>" />
            </label>
            <label>
              <span>eos_token</span>
              <input className="input mono sm" value={state.pasteEos} onChange={(e) => set({ pasteEos: e.target.value })} placeholder="</s>" />
            </label>
          </div>
        )}
      </section>

      <div className="lab-grid">
        {/* ------------------------------------------------ editor */}
        <section className="col" aria-label="Conversation">
          <div className="card">
            <div className="card-head">
              <h2>Conversation</h2>
              <span className="muted small">{state.msgs.length} messages</span>
            </div>
            <ol className="msgs">
              {state.msgs.map((m, i) => (
                <li key={i} className={`msg role-${m.role}`}>
                  <div className="msg-head">
                    <label className="role-select">
                      <span className="sr-only">Role of message {i + 1}</span>
                      <select value={m.role} onChange={(e) => updateMsg(i, { role: e.target.value as Role })}>
                        <option value="system">system</option>
                        <option value="user">user</option>
                        <option value="assistant">assistant</option>
                        <option value="tool">tool</option>
                      </select>
                    </label>
                    <span className="msg-actions">
                      {m.role === 'assistant' && (
                        <button className="mini" onClick={() => updateMsg(i, { toolCalls: m.toolCalls === undefined ? JSON.stringify([{ id: 'abc123xyz', type: 'function', function: { name: 'get_weather', arguments: { city: 'Paris' } } }], null, 2) : undefined })}>
                          {m.toolCalls === undefined ? '+ tool_calls' : '− tool_calls'}
                        </button>
                      )}
                      <button className="mini icon" aria-label="Move up" disabled={i === 0} onClick={() => moveMsg(i, -1)}>↑</button>
                      <button className="mini icon" aria-label="Move down" disabled={i === state.msgs.length - 1} onClick={() => moveMsg(i, 1)}>↓</button>
                      <button className="mini icon danger" aria-label={`Delete message ${i + 1}`} onClick={() => set({ msgs: state.msgs.filter((_, j) => j !== i) })}>×</button>
                    </span>
                  </div>
                  <textarea className="msg-text" aria-label={`Message ${i + 1} content`} value={m.content} onChange={(e) => updateMsg(i, { content: e.target.value })} rows={Math.min(8, Math.max(1, m.content.split('\n').length))} placeholder={m.role === 'assistant' && m.toolCalls ? '(empty: tool call only)' : `${m.role} message…`} />
                  {m.role === 'assistant' && m.toolCalls !== undefined && (
                    <textarea className="msg-text mono small-code" aria-label={`Message ${i + 1} tool_calls JSON`} value={m.toolCalls} onChange={(e) => updateMsg(i, { toolCalls: e.target.value })} rows={6} spellCheck={false} />
                  )}
                  {m.role === 'tool' && (
                    <label className="inline-field">
                      <span>tool_call_id</span>
                      <input className="input mono sm" value={m.toolCallId ?? ''} onChange={(e) => updateMsg(i, { toolCallId: e.target.value })} />
                    </label>
                  )}
                </li>
              ))}
            </ol>
            <div className="add-row">
              {(['system', 'user', 'assistant', 'tool'] as Role[]).map((r) => (
                <button key={r} className={`btn btn-sm add-${r}`} onClick={() => addMsg(r)}>
                  + {r}
                </button>
              ))}
            </div>
          </div>

          <div className="card">
            <div className="card-head">
              <h2>Options</h2>
            </div>
            <label className="switch">
              <input type="checkbox" checked={state.agp} onChange={(e) => set({ agp: e.target.checked })} />
              <span className="track" aria-hidden />
              <span>
                <code>add_generation_prompt</code>
                <small className="muted">Append the assistant header for inference</small>
              </span>
            </label>
            <label className="switch">
              <input type="checkbox" checked={state.toolsOn} onChange={(e) => set({ toolsOn: e.target.checked })} />
              <span className="track" aria-hidden />
              <span>
                <code>tools</code>
                <small className="muted">JSON schema list passed to the template</small>
              </span>
            </label>
            {state.toolsOn && <textarea className="code-area" aria-label="Tools JSON" value={state.tools} onChange={(e) => set({ tools: e.target.value })} rows={10} spellCheck={false} />}
            <div className="field">
              <div className="field-head">
                <label htmlFor="kwargs">Extra kwargs</label>
                {kwargsDetected.length > 0 && (
                  <span className="detected">
                    <span className="muted small">template reads:</span>
                    {kwargsDetected.map((k) => (
                      <button key={k} className="pill sm" onClick={() => addKwarg(k)} title={`Add ${k} to kwargs`}>
                        {k}
                      </button>
                    ))}
                  </span>
                )}
              </div>
              <textarea id="kwargs" className="code-area" value={state.kwargs} onChange={(e) => set({ kwargs: e.target.value })} rows={3} spellCheck={false} />
            </div>
            {errors.length > 0 && (
              <ul className="input-errors" role="alert">
                {errors.map((e) => (
                  <li key={e}>{e}</li>
                ))}
              </ul>
            )}
          </div>
        </section>

        {/* ------------------------------------------------ output */}
        <section className="col" aria-label="Output">
          <div className="card lint">
            <div className="card-head">
              <h2>Linter</h2>
              {counts && (
                <span className="counts">
                  {counts.error > 0 && <span className="count sev-error">{counts.error} error{counts.error > 1 ? 's' : ''}</span>}
                  {counts.warn > 0 && <span className="count sev-warn">{counts.warn} warning{counts.warn > 1 ? 's' : ''}</span>}
                  {counts.error + counts.warn === 0 && <span className="count sev-ok">clean</span>}
                </span>
              )}
            </div>
            {A ? <IssueList issues={A.issues} /> : <Placeholder hub={hubA} />}
          </div>

          <div className="card out">
            <div className="tabs" role="tablist" aria-label="Output views">
              {(['rendered', 'tokens', 'compare', 'template'] as const).map((t) => (
                <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)}>
                  {t === 'rendered' ? 'Rendered' : t === 'tokens' ? 'Tokens' : t === 'compare' ? 'Compare models' : 'Template'}
                  {t === 'tokens' && tokensA && <span className="tab-badge">{tokensA.ids.length.toLocaleString()}</span>}
                </button>
              ))}
            </div>

            {tab === 'rendered' && (
              <div className="pane">
                <div className="pane-bar">
                  <span className="muted small">{A?.result.ok ? `${A.result.text.length.toLocaleString()} chars${tokensA ? ` · ${tokensA.ids.length.toLocaleString()} tokens` : ''}` : ''}</span>
                  <span className="pane-tools">
                    <label className="check"><input type="checkbox" checked={showWs} onChange={(e) => setShowWs(e.target.checked)} /> whitespace</label>
                    {A?.result.ok && <button className="btn btn-sm" onClick={() => { navigator.clipboard.writeText((A.result as { text: string }).text); flash('Rendered prompt copied') }}>Copy</button>}
                  </span>
                </div>
                {A ? <Output built={A} showWs={showWs} /> : <Placeholder hub={hubA} />}
              </div>
            )}

            {tab === 'tokens' && (
              <div className="pane">
                <TokenizerBar tok={tokA} id={tokIdA} on={state.tokenize} onToggle={(v) => set({ tokenize: v })} paste={mode === 'paste'} />
                {tokensA && A && (
                  <>
                    <div className="pane-bar">
                      <span className="stat-row">
                        <span className="stat"><b>{tokensA.ids.length.toLocaleString()}</b> tokens</span>
                        <span className="stat"><b>{tokensA.pieces.filter((p) => A.specialList.includes(p)).length}</b> special</span>
                        <span className="stat"><b>{A.result.ok ? (A.result.text.length / Math.max(1, tokensA.ids.length)).toFixed(2) : '–'}</b> chars/token</span>
                      </span>
                      <label className="check"><input type="checkbox" checked={showIds} onChange={(e) => setShowIds(e.target.checked)} /> ids</label>
                    </div>
                    <TokenChips ids={tokensA.ids} pieces={tokensA.pieces} specials={new Set(A.specialList)} showIds={showIds} />
                  </>
                )}
              </div>
            )}

            {tab === 'compare' && (
              <div className="pane">
                <form className="model-form" onSubmit={(e) => { e.preventDefault(); set({ compare: compareDraft.trim() }) }}>
                  <label className="sr-only" htmlFor="compare-id">Compare with model</label>
                  <div className="input-icon">
                    <span className="vs">vs</span>
                    <input id="compare-id" className="input mono" value={compareDraft} onChange={(e) => setCompareDraft(e.target.value)} placeholder="Second model id" spellCheck={false} />
                  </div>
                  <button className="btn btn-dark" type="submit">Compare</button>
                </form>
                <div className="quick">
                  {DEFAULT_MODELS.filter((m) => m !== state.model).slice(0, 6).map((id) => (
                    <button key={id} className={`pill ${state.compare === id ? 'on' : ''}`} onClick={() => { setCompareDraft(id); set({ compare: id }) }}>
                      {id.split('/')[1]}
                    </button>
                  ))}
                </div>
                {hubB.status === 'error' && <HubErrorCard error={hubB.error} onPick={(id) => { setCompareDraft(id); set({ compare: id }) }} />}
                {hubB.status === 'loading' && <div className="skeleton" />}
                {state.tokenize && [tokA, tokB].map((t, i) => t.status === 'error' && <p key={i} className="err-text small">{t.message}</p>)}
                {A && B && <Compare a={A} b={B} nameA={state.model || 'Pasted template'} nameB={state.compare} tokA={tokensA?.ids.length} tokB={tokensB?.ids.length} showWs={showWs} tokenizeOn={state.tokenize} onTokenize={() => set({ tokenize: true })} />}
              </div>
            )}

            {tab === 'template' && (
              <div className="pane">
                <div className="pane-bar">
                  <span className="muted small">
                    {mode === 'paste' ? 'Paste any Jinja chat template. It renders as you type.' : state.override !== null ? 'Edited locally. The linter and render use your version.' : modelA ? `From ${modelA.source}. Edit to experiment.` : ''}
                  </span>
                  {mode === 'hub' && state.override !== null && <button className="btn btn-sm" onClick={() => set({ override: null })}>Reset to Hub version</button>}
                </div>
                <textarea
                  className="code-area template-area"
                  aria-label="Chat template source"
                  value={(mode === 'paste' ? state.override : state.override ?? hubTemplate) ?? ''}
                  onChange={(e) => set({ override: e.target.value })}
                  placeholder={"{% for message in messages %}…{% endfor %}"}
                  spellCheck={false}
                />
              </div>
            )}
          </div>
        </section>
      </div>
      <div className={`toast ${toast ? 'show' : ''}`} role="status" aria-live="polite">
        {toast}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------- pieces

function ModelMeta({ hub, onPick, override, onReset, tplName, onTpl }: { hub: HubState; onPick: (id: string) => void; override: boolean; onReset: () => void; tplName: string; onTpl: (n: string) => void }) {
  if (hub.status === 'loading') return <div className="meta"><span className="spinner" /> Fetching tokenizer_config.json…</div>
  if (hub.status === 'error') return <HubErrorCard error={hub.error} onPick={onPick} />
  if (hub.status !== 'ready') return null
  const m = hub.model
  return (
    <div className="meta">
      <a className="meta-id mono" href={`https://huggingface.co/${m.id}`} target="_blank" rel="noreferrer">{m.id}</a>
      <span className={`tag ${m.source === 'none' ? 'tag-bad' : ''}`}>{m.source === 'none' ? 'no chat_template' : m.source}</span>
      {m.templates.length > 1 && (
        <label className="select-wrap sm">
          <span className="sr-only">Named template</span>
          <select value={tplName} onChange={(e) => onTpl(e.target.value)}>
            {m.templates.map((t) => <option key={t.name} value={t.name}>{t.name}</option>)}
          </select>
        </label>
      )}
      <span className="tag mono">bos {m.specials.bos_token ?? '–'}</span>
      <span className="tag mono">eos {m.specials.eos_token ?? '–'}</span>
      <span className="tag mono">add_bos_token {m.addBosToken === null ? 'unset' : String(m.addBosToken)}</span>
      {override && <button className="tag tag-warn" onClick={onReset} title="Discard local edits">edited · reset</button>}
    </div>
  )
}

function HubErrorCard({ error, onPick }: { error: HubError; onPick: (id: string) => void }) {
  const picks = [...new Set([...error.suggestions, ...(error.kind === 'gated' || error.kind === 'not-found' ? DEFAULT_MODELS.slice(0, 4) : [])])]
  return (
    <div className={`hub-error kind-${error.kind}`} role="alert">
      <div>
        <strong>{error.kind === 'gated' ? 'Gated model' : error.kind === 'not-found' ? 'Model not found' : error.kind === 'invalid' ? 'Not a model id' : 'Network problem'}</strong>
        <p>{error.message}</p>
        {error.kind === 'gated' && <p className="muted small">You can still paste its template: copy chat_template from the repo page (after accepting the license) into “Paste a template”.</p>}
      </div>
      {picks.length > 0 && (
        <div className="quick">
          <span className="muted small">{error.kind === 'gated' ? 'Try an ungated copy:' : 'Try:'}</span>
          {picks.map((id) => (
            <button key={id} className={`pill ${id === error.suggestions[0] ? 'on' : ''}`} onClick={() => onPick(id)}>{id}</button>
          ))}
        </div>
      )}
    </div>
  )
}

function Placeholder({ hub }: { hub: HubState }) {
  if (hub.status === 'loading') return <div className="skeleton" />
  if (hub.status === 'error') return <p className="muted small">Load a model or paste a template to see output.</p>
  return <div className="skeleton" />
}

function Output({ built, showWs }: { built: Built; showWs: boolean }) {
  const r = built.result
  if (!r.ok)
    return (
      <div className={`render-error kind-${r.error.kind}`} role="alert">
        <span className="badge">{r.error.kind === 'raise' ? 'raise_exception' : r.error.kind === 'parse' ? 'syntax error' : 'runtime error'}</span>
        <p className="mono">{r.error.message}</p>
        <p className="muted small">{r.error.kind === 'raise' ? 'The template itself rejected this conversation. transformers would throw TemplateError here too.' : 'This is what apply_chat_template() would fail with.'}</p>
      </div>
    )
  if (!r.text) return <p className="muted small">The template rendered an empty string.</p>
  return <Rendered text={r.text} specials={built.specialList} showWs={showWs} />
}

function TokenizerBar({ tok, id, on, onToggle, paste }: { tok: TokState; id: string; on: boolean; onToggle: (v: boolean) => void; paste: boolean }) {
  if (!on)
    return (
      <div className="tok-cta">
        <div>
          <strong>Tokenize with the real tokenizer</strong>
          <p className="muted small">
            Loads <code>{id}</code>’s tokenizer.json with transformers.js (a few MB, cached by your browser).{paste ? ' Pasted templates use the Qwen2.5 tokenizer.' : ''} Shows exact ids, counts, and whether encode() adds its own BOS.
          </p>
        </div>
        <button className="btn btn-yellow" onClick={() => onToggle(true)}>Load tokenizer</button>
      </div>
    )
  return (
    <div className="pane-bar">
      <span className="small">
        {tok.status === 'loading' && <><span className="spinner" /> Loading {id} tokenizer…</>}
        {tok.status === 'ready' && <span className="muted">Tokenizer: <code>{id}</code> · add_special_tokens=False (as apply_chat_template does)</span>}
        {tok.status === 'error' && <span className="err-text">{tok.message}</span>}
      </span>
      <button className="btn btn-sm" onClick={() => onToggle(false)}>Turn off</button>
    </div>
  )
}

function Compare({ a, b, nameA, nameB, tokA, tokB, showWs, tokenizeOn, onTokenize }: { a: Built; b: Built; nameA: string; nameB: string; tokA?: number; tokB?: number; showWs: boolean; tokenizeOn: boolean; onTokenize: () => void }) {
  const ta = a.result.ok ? a.result.text : null
  const tb = b.result.ok ? b.result.text : null
  const ops = useMemo(() => (ta !== null && tb !== null ? diff(diffUnits(ta, a.specialList), diffUnits(tb, b.specialList)) : null), [ta, tb, a.specialList, b.specialList])
  const sa = summarize(a.issues)
  const sb = summarize(b.issues)
  const delta = tokA !== undefined && tokB !== undefined ? tokB - tokA : null
  const sides = [
    { name: nameA, built: a, text: ta, tokens: tokA, s: sa, hide: 'add' as const, mark: 'del' },
    { name: nameB, built: b, text: tb, tokens: tokB, s: sb, hide: 'del' as const, mark: 'add' },
  ]
  return (
    <>
      <div className="cmp-stats">
        {sides.map((x, i) => (
          <div key={i} className={`cmp-card side-${i}`}>
            <span className="mono small cmp-name">{x.name}</span>
            <span className="cmp-nums">
              <b>{x.tokens !== undefined ? x.tokens.toLocaleString() : x.text !== null ? `≈${Math.round(x.text.length / 3.6).toLocaleString()}` : '–'}</b> tokens
              <span className="muted"> · {x.text?.length.toLocaleString() ?? '–'} chars</span>
            </span>
            <span className="counts">
              {x.s.error > 0 && <span className="count sev-error">{x.s.error} err</span>}
              {x.s.warn > 0 && <span className="count sev-warn">{x.s.warn} warn</span>}
              {x.s.error + x.s.warn === 0 && <span className="count sev-ok">clean</span>}
            </span>
          </div>
        ))}
        <div className="cmp-delta">
          {delta !== null ? (
            <><b className={delta > 0 ? 'up' : delta < 0 ? 'down' : ''}>{delta > 0 ? '+' : ''}{delta}</b><span className="muted small">tokens for B ({tokA ? ((delta / tokA) * 100).toFixed(0) : 0}%)</span></>
          ) : ta === null || tb === null ? (
            <span className="muted small">No delta: a template raised</span>
          ) : (
            <button className="btn btn-sm" onClick={onTokenize} disabled={tokenizeOn}>{tokenizeOn ? 'Tokenizing…' : 'Exact token counts'}</button>
          )}
        </div>
      </div>
      <div className="cmp-grid">
        {sides.map((x, i) => (
          <div key={i} className="cmp-col">
            {x.text === null || !ops ? (
              <Output built={x.built} showWs={showWs} />
            ) : (
              <pre className="rendered">
                {ops.filter((o) => o.op !== x.hide).map((o, j) => (
                  <span key={j} className={o.op === 'same' ? undefined : `d-${x.mark}`}>{highlight(o.text, x.built.specialList, showWs)}</span>
                ))}
              </pre>
            )}
            <IssueList issues={x.built.issues.filter((s) => s.severity === 'error' || s.severity === 'warn')} compact />
          </div>
        ))}
      </div>
    </>
  )
}

