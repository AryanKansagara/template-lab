import { useEffect, useMemo, useState } from 'react'
import { lint, render, segment, type ChatMessage } from './lib/template.ts'
import { HERO_TEMPLATES } from './samples.ts'
import { Link } from './App.tsx'
import { chipText, highlight } from './views.tsx'

const SYSTEM = 'Be brief.'
const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches

/** Word-level split for the hero only; the lab uses the model's real tokenizer. */
function roughTokens(text: string) {
  return segment(text).flatMap((s) => (s.special ? [{ text: s.text, special: true }] : (s.text.match(/\n+|\s*[\w']+|\s*[^\s\w]/g) ?? []).map((t) => ({ text: t, special: false }))))
}

function HeroDemo() {
  const [idx, setIdx] = useState(0)
  const [auto, setAuto] = useState(() => !reduced())
  const [user, setUser] = useState('Why is my model rambling?')
  const [typed, setTyped] = useState(() => (reduced() ? Infinity : 0))

  const t = HERO_TEMPLATES[idx]
  const messages: ChatMessage[] = useMemo(() => [{ role: 'system', content: SYSTEM }, { role: 'user', content: user }], [user])
  const input = { messages, addGenerationPrompt: true, specials: { bos_token: t.bos, eos_token: t.bos ? '</s>' : '<|im_end|>' } }
  const result = render({ ...input, template: t.template })
  const text = result.ok ? result.text : ''
  const flagged = lint({ template: t.template, input, tokenizerAddsBos: t.bos ? true : false }).filter((i) => i.severity === 'error' || i.severity === 'warn')
  const tokens = useMemo(() => roughTokens(text), [text])
  const done = typed >= text.length
  // Chips appear as the typewriter passes them, so the render visibly turns into tokens.
  let end = 0
  const visible = tokens.filter((tk) => (end += tk.text.length) <= typed)

  // Typewriter, then hold on the chips, then next model.
  useEffect(() => {
    if (!auto) return
    const timer = done ? setTimeout(() => { setIdx((i) => (i + 1) % HERO_TEMPLATES.length); setTyped(0) }, 4200) : setTimeout(() => setTyped((n) => n + 3), 22)
    return () => clearTimeout(timer)
  }, [auto, done, typed])

  const pick = (i: number) => {
    setIdx(i)
    setAuto(false)
    setTyped(Infinity)
  }

  return (
    <div className="hero-demo" aria-label="Live chat template render">
      <div className="demo-tabs" role="tablist" aria-label="Template">
        {HERO_TEMPLATES.map((h, i) => (
          <button key={h.model} role="tab" aria-selected={i === idx} onClick={() => pick(i)}>
            {h.model.split('/')[1]}
          </button>
        ))}
      </div>
      <div className="demo-chat">
        <div className="bubble role-system"><b>system</b> {SYSTEM}</div>
        <label className="bubble role-user">
          <b>user</b>
          <input aria-label="Try your own user message" value={user} onChange={(e) => { setUser(e.target.value); setAuto(false); setTyped(Infinity) }} spellCheck={false} />
        </label>
      </div>
      <div className="demo-flow" aria-hidden>apply_chat_template</div>
      <pre className="rendered demo-render">
        {highlight(text.slice(0, typed), [], true)}
        {!done && <span className="caret" />}
      </pre>
      <div className="demo-tokens">
        <div className="demo-chips">
          {visible.slice(0, 44).map((tk, i) => (
            <span key={`${idx}-${i}`} className={`chip ${tk.special ? 'chip-sp' : `c${i % 5}`}`}>
              {chipText(tk.text)}
            </span>
          ))}
        </div>
        <div className="demo-foot">
          <span className="mono small">{visible.length} word pieces · {Math.min(typed, text.length)} chars</span>
          {!done ? <span className="muted small">rendering…</span> : flagged.length ? <span className="count sev-warn">{flagged[0].title}</span> : <span className="count sev-ok">linter clean</span>}
        </div>
      </div>
    </div>
  )
}

const CASES = [
  { model: 'unsloth/Llama-3.1-8B-Instruct', sample: 'chat', tag: 'Double BOS', sev: 'error', body: 'The template writes <|begin_of_text|> and add_bos_token is true. Render, then encode with defaults, and the model silently sees two.' },
  { model: 'microsoft/Phi-3.5-mini-instruct', sample: 'tools', tag: 'Tools ignored', sev: 'warn', body: 'Pass tools=[…] and the render is byte-identical. No error, and the model never learns the functions exist.' },
  { model: 'mistralai/Mistral-7B-Instruct-v0.3', sample: 'chat', tag: 'System prompt moves', sev: 'warn', body: 'It only renders the system prompt on the final user turn. End on an assistant message and it vanishes, so your training prefix never matches inference.' },
  { model: 'unsloth/gemma-2-2b-it', sample: 'chat', tag: 'raise_exception', sev: 'error', body: 'A system message throws “System role not supported” at runtime. See the error before your users do.' },
]

const FEATURES = [
  ['Any Hub repo', 'Reads tokenizer_config.json, chat_template.jinja or chat_template.json, including named templates. Gated repos get a clear explanation and a verified ungated copy.'],
  ['Renders as you type', 'The official @huggingface/jinja engine, with system, user, assistant and tool turns, tool schemas, add_generation_prompt and custom kwargs.'],
  ['Nothing invisible', 'Special tokens highlighted, spaces and newlines drawn. The difference between "\\n" and "\\n\\n" stops being a mystery.'],
  ['Real token ids', 'Lazy-loads the model’s own tokenizer with transformers.js for exact ids and counts, and checks whether encode() adds its own BOS.'],
  ['Compare two models', 'One conversation, two templates, side by side: a diff of the prompts plus the token delta you pay for switching.'],
  ['Links that reproduce', 'The whole setup compresses into the URL. Paste it into the issue instead of describing what you saw.'],
]

export default function Landing() {
  return (
    <div className="landing">
      <section className="hero">
        <div className="hero-copy">
          <span className="eyebrow">For anyone shipping a chat template</span>
          <h1>See the prompt your model <mark>actually</mark> gets.</h1>
          <p>
            Chat templates fail quietly: a dropped system prompt, a second BOS, a stray newline. Your model just gets a little worse. Template Lab renders any Hub model’s <code>chat_template</code> live and tells you what went wrong.
          </p>
          <div className="hero-cta">
            <Link className="btn btn-dark btn-lg" href="/app">Open the lab</Link>
            <Link className="btn btn-lg" href="/app#try=tools|Qwen%2FQwen2.5-7B-Instruct">Try a tool-call render</Link>
          </div>
          <p className="muted small">Free, no login. Runs in your browser; only public model files are fetched.</p>
        </div>
        <HeroDemo />
      </section>

      <section className="band">
        <h2>Four real repos, four quiet problems</h2>
        <p className="band-sub">Each of these comes from the repo’s current config. Open one and watch the linter catch it.</p>
        <div className="case-grid">
          {CASES.map((c) => (
            <Link key={c.model} className="model-card" href={`/app#try=${c.sample}|${encodeURIComponent(c.model)}`}>
              <span className="mono model-name">{c.model}</span>
              <span className={`count sev-${c.sev}`}>{c.tag}</span>
              <p>{c.body}</p>
              <span className="card-go">Open in the lab →</span>
            </Link>
          ))}
        </div>
      </section>

      <section className="band">
        <h2>Everything <code>apply_chat_template</code> hides</h2>
        <div className="feature-grid">
          {FEATURES.map(([title, body]) => (
            <div key={title} className="feature">
              <h3>{title}</h3>
              <p>{body}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="closing">
        <h2>Paste the model id you’re debugging right now.</h2>
        <Link className="btn btn-yellow btn-lg" href="/app">Open the lab</Link>
      </section>
    </div>
  )
}
