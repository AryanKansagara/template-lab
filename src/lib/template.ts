import { Template } from '@huggingface/jinja'
import { floatJson, floatRepr, PyFloat } from './json.ts'

export type Role = 'system' | 'user' | 'assistant' | 'tool'

/** A message as the template sees it (HF chat format). */
export interface ChatMessage {
  role: Role
  content: string
  tool_calls?: unknown[]
  tool_call_id?: string
  [k: string]: unknown
}

export interface SpecialTokens {
  bos_token?: string
  eos_token?: string
  unk_token?: string
  pad_token?: string
}

export interface RenderInput {
  template: string
  messages: ChatMessage[]
  tools?: unknown[] | null
  addGenerationPrompt: boolean
  kwargs?: Record<string, unknown>
  specials?: SpecialTokens
}

/** Nothing was rendered for `input` (the tools, kwargs or tool_calls JSON can't be read) and `choice` (no named template applies). */
export type RenderError = { kind: 'parse' | 'raise' | 'runtime' | 'input' | 'choice'; message: string }
export type RenderResult = { ok: true; text: string } | { ok: false; error: RenderError }

/** What transformers < 4.43 silently used when a repo had no chat_template. */
export const CHATML_FALLBACK =
  "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

class RaiseError extends Error {}

const pyQuote = (s: string) => {
  const q = s.includes("'") && !s.includes('"') ? '"' : "'"
  return q + s.replace(/\\/g, '\\\\').replace(new RegExp(q, 'g'), '\\' + q).replace(/\n/g, '\\n').replace(/\t/g, '\\t').replace(/\r/g, '\\r') + q
}

/** Python repr() of a value as the jinja runtime hands it over (Map for dicts, runtime values inside). */
export function pyRepr(v: unknown): string {
  if (v instanceof Map) return '{' + [...v].map(([k, x]) => `${pyRepr(k)}: ${pyRepr(x)}`).join(', ') + '}'
  if (Array.isArray(v)) return '[' + v.map(pyRepr).join(', ') + ']'
  if (v && typeof v === 'object' && 'type' in v && 'value' in v) {
    const r = v as { type: string; value: unknown }
    if (r.type === 'FloatValue') return floatRepr(r.value as number)
    return pyRepr(r.value)
  }
  if (typeof v === 'string') return pyQuote(v)
  if (typeof v === 'boolean') return v ? 'True' : 'False'
  if (v == null) return 'None'
  if (v && typeof v === 'object') return pyRepr(new Map(Object.entries(v)))
  return String(v)
}

/** Python str() of a one-item list: inside a list the engine keeps the runtime type (1.0, undefined), which it strips from plain arguments. */
const pyStr = ([v]: { type: string; value: unknown }[]) => (v.type === 'UndefinedValue' ? '' : v.type === 'StringValue' ? v.value : pyRepr(v))

type Runtime = { type: string; value: unknown }
interface JsonOptions { indent?: number | null; ensureAscii?: boolean; separators?: [string, string] | null; sortKeys?: boolean }

/** json.dumps() of a runtime value. The engine's own tojson writes every float as a JS number. */
function pyJson(v: Runtime, o: JsonOptions, depth = 0): string {
  const str = (s: unknown) => {
    const j = JSON.stringify(s)
    return o.ensureAscii ? j.replace(/[\u007f-\uffff]/g, (c) => '\\u' + c.charCodeAt(0).toString(16).padStart(4, '0')) : j
  }
  const [item, key] = o.separators ?? [o.indent ? ',' : ', ', ': ']
  const wrap = (open: string, parts: string[], close: string) => {
    if (!o.indent || !parts.length) return open + parts.join(item) + close
    const pad = '\n' + ' '.repeat(o.indent * depth)
    const inner = pad + ' '.repeat(o.indent)
    return open + inner + parts.join(item + inner) + pad + close
  }
  switch (v.type) {
    case 'NullValue':
    case 'UndefinedValue':
      return 'null'
    case 'FloatValue':
      return floatJson(v.value as number)
    case 'IntegerValue':
    case 'BooleanValue':
      return String(v.value)
    case 'StringValue':
      return str(v.value)
    case 'ArrayValue':
    case 'TupleValue':
      return wrap('[', (v.value as Runtime[]).map((x) => pyJson(x, o, depth + 1)), ']')
    case 'ObjectValue':
    case 'NamespaceValue': {
      const entries = [...(v.value as Map<string, Runtime>)]
      if (o.sortKeys) entries.sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
      return wrap('{', entries.map(([k, x]) => str(k) + key + pyJson(x, o, depth + 1)), '}')
    }
    default:
      throw new Error(`Cannot convert to JSON: ${v.type}`)
  }
}

/** The tojson filter, called as ([operand], keyword arguments). */
function toJson([v]: Runtime[], named?: unknown): string {
  const kw = named instanceof Map ? (named as Map<string, Runtime>) : new Map<string, Runtime>()
  const option = <T>(name: string, type: string, what: string) => {
    const x = kw.get(name)
    if (!x || x.type === 'NullValue') return undefined
    if (x.type !== type) throw new Error(`If set, ${name} must be ${what}`)
    return x.value as T
  }
  const sep = kw.get('separators')
  const pair = sep && sep.type !== 'NullValue' ? (sep.value as Runtime[]) : null
  if (pair && !(Array.isArray(pair) && pair.length === 2 && pair.every((s) => s.type === 'StringValue'))) throw new Error('If set, separators must be a tuple of two strings')
  return pyJson(v, {
    indent: option<number>('indent', 'IntegerValue', 'a number'),
    ensureAscii: option<boolean>('ensure_ascii', 'BooleanValue', 'a boolean'),
    sortKeys: option<boolean>('sort_keys', 'BooleanValue', 'a boolean'),
    separators: pair && [pair[0].value as string, pair[1].value as string],
  })
}

type Node = { type?: string; value?: unknown; callee?: Node; filter?: Node; operand?: Node; args?: unknown[]; [k: string]: unknown }

/** A call of our function `fn` on an expression, boxed in a list so that its runtime type survives. */
const callOn = (fn: string, operand: unknown, args: unknown[] = []) => ({ type: 'CallExpression', callee: { type: 'Identifier', value: fn }, args: [{ type: 'ArrayLiteral', value: [operand] }, ...args] })

/** `join`, unless it picks an attribute of the items. */
const JOINS = (f: Node) => (f.type === 'CallExpression' ? f.callee : f)?.value === 'join' && !(f.args as Node[] | undefined)?.some((a) => (a.key as Node | undefined)?.value === 'attribute')

/** The items of a list as strings; any other value as the engine would see it. */
const pyItems = ([v]: Runtime[]) => (Array.isArray(v.value) ? (v.value as Runtime[]).map((x) => pyStr([x])) : v.type === 'StringValue' ? v.value : null)

/** Does a value hold a number JS can't pass to the engine as it is? */
const hasPyNumber = (v: unknown): boolean =>
  typeof v === 'bigint' || v instanceof PyFloat || (!!v && typeof v === 'object' && Object.values(v).some(hasPyNumber))

/** A value as a template literal, the only way to hand the engine a whole-valued float or a large integer. */
function literal(v: unknown): Node {
  if (v instanceof PyFloat) return { type: 'FloatLiteral', value: v.value }
  if (typeof v === 'bigint') return { type: 'IntegerLiteral', value: v }
  if (typeof v === 'number') return { type: Number.isInteger(v) ? 'IntegerLiteral' : 'FloatLiteral', value: v }
  if (typeof v === 'string') return { type: 'StringLiteral', value: v }
  if (Array.isArray(v)) return { type: 'ArrayLiteral', value: v.map(literal) }
  if (v && typeof v === 'object') return { type: 'ObjectLiteral', value: new Map(Object.entries(v).map(([k, x]) => [literal(k), literal(x)])) }
  return { type: 'Identifier', value: v === undefined ? '__tl_undefined' : v === null ? '__tl_none' : v ? '__tl_true' : '__tl_false' }
}

// {{ x }} of a float, list or dict prints repr(x); the engine prints JSON with JS numbers.
// Its classes are not exported, so they are taken from values.
new Template('{{ take([0.5, [], {}]) }}').render({
  take: (values: Runtime[]) =>
    values.forEach((v) => {
      v.constructor.prototype.toString = function (this: Runtime) {
        return pyRepr(this)
      }
    }),
})

/**
 * Route `x | string`, `x | tojson`, the operands of `~`, the items of `x | join` and `raise_exception(...)` to our own functions, on the parsed template so that plain text,
 * literals and comments that merely look like them are left alone.
 */
function hook(x: unknown): void {
  if (Array.isArray(x)) return x.forEach(hook)
  if (x instanceof Map) return x.forEach((v, k) => (hook(k), hook(v)))
  if (!x || typeof x !== 'object') return
  const n = x as Node
  Object.values(n).forEach(hook)
  if (n.type === 'CallExpression' && n.callee?.type === 'Identifier' && n.callee.value === 'raise_exception') n.callee.value = '__tl_raise'
  // `x | string` becomes __tl_str(x). The filter takes no arguments.
  if (n.type === 'FilterExpression') {
    const f = n.filter!
    const name = f.type === 'CallExpression' ? f.callee : f
    if (name?.type === 'Identifier' && (name.value === 'tojson' || (name.value === 'string' && !f.args?.length))) {
      const call = callOn(name.value === 'string' ? '__tl_str' : '__tl_json', n.operand, (f.args ?? []).filter((a) => (a as Node).type?.startsWith('Keyword')))
      delete n.operand
      delete n.filter
      Object.assign(n, call)
    }
  }
  // `join` joins str() of the items, like `~` below.
  if (n.type === 'FilterExpression' && JOINS(n.filter!)) n.operand = callOn('__tl_items', n.operand)
  // `~` joins str() of both sides; the engine joins JS values, so 1.0 would become 1.
  if (n.type === 'BinaryExpression' && (n.operator as Node | undefined)?.value === '~') {
    n.left = callOn('__tl_str', n.left)
    n.right = callOn('__tl_str', n.right)
  }
}

const compiled = new Map<string, Template | RenderError>()

function compile(src: string): Template | RenderError {
  let t = compiled.get(src)
  if (!t) {
    try {
      // raise_exception is hooked to tell it apart from crashes (the builtin can't be overridden);
      // `x | string` because it is Python's repr in transformers, which @huggingface/jinja can't do for objects.
      t = new Template(src)
      hook(t.parsed)
    } catch (e) {
      t = { kind: 'parse', message: (e as Error).message }
    }
    if (compiled.size > 50) compiled.clear()
    compiled.set(src, t)
  }
  return t
}

export function render(input: RenderInput): RenderResult {
  const t = compile(input.template)
  if (!(t instanceof Template)) return { ok: false, error: t }
  try {
    const context: Record<string, unknown> = {
      ...input.specials,
      ...input.kwargs,
      messages: input.messages,
      tools: input.tools?.length ? input.tools : null,
      add_generation_prompt: input.addGenerationPrompt,
    }
    const odd = Object.entries(context).filter(([, v]) => hasPyNumber(v))
    const program = Object.create(t) as Template
    if (odd.length) {
      const sets = odd.map(([k, v]) => ({ type: 'Set', assignee: { type: 'Identifier', value: k }, value: literal(v), body: [] }))
      program.parsed = { ...t.parsed, body: [...(sets as unknown as typeof t.parsed.body), ...t.parsed.body] }
    }
    const text = program.render({
      ...context,
      ...Object.fromEntries(odd.map(([k]) => [k, null])),
      __tl_undefined: undefined,
      __tl_none: null,
      __tl_true: true,
      __tl_false: false,
      __tl_str: pyStr,
      __tl_json: toJson,
      __tl_items: pyItems,
      __tl_raise: (msg: unknown) => {
        throw new RaiseError(String(msg))
      },
    })
    return { ok: true, text }
  } catch (e) {
    const { message } = e as Error
    if (e instanceof TypeError && /BigInt/i.test(message)) {
      return { ok: false, error: { kind: 'runtime', message: 'The template calculates with an integer beyond ±9,007,199,254,740,991, which this lab can print but not compute with exactly. Python would not fail here.' } }
    }
    return { ok: false, error: { kind: e instanceof RaiseError ? 'raise' : 'runtime', message } }
  }
}

// ---------------------------------------------------------------- linter

export type Severity = 'error' | 'warn' | 'info' | 'ok'

export interface Issue {
  id: string
  severity: Severity
  title: string
  detail: string
}

export interface LintContext {
  /** null = repo has no chat_template at all */
  template: string | null
  input: Omit<RenderInput, 'template'>
  /** Does the tokenizer prepend BOS on encode()? null = unknown. */
  tokenizerAddsBos: boolean | null
}

const rank: Record<Severity, number> = { error: 0, warn: 1, info: 2, ok: 3 }

/** Marker that survives trim/split/startswith logic in real templates. */
const marker = (i: number) => `⟦m${i}⟧`

export function lint(ctx: LintContext): Issue[] {
  const issues: Issue[] = []
  const add = (severity: Severity, id: string, title: string, detail: string) => issues.push({ id, severity, title, detail })
  const hasTemplate = !!ctx.template?.trim()
  const tpl = hasTemplate ? ctx.template! : CHATML_FALLBACK
  const base = ctx.input
  const run = (over: Partial<RenderInput>) => render({ ...base, template: tpl, ...over })

  if (!hasTemplate) {
    add(
      'error',
      'missing-template',
      'No chat template',
      'transformers ≥ 4.43 raises "Cannot use chat template functions because tokenizer.chat_template is not set". Older versions silently fell back to ChatML, which this model was probably never trained on. Shown below: the ChatML fallback.',
    )
  }

  const main = run({})
  if (!main.ok) {
    const { kind, message } = main.error
    if (kind === 'raise') {
      let hint = 'The template refuses this conversation shape.'
      if (/alternat/i.test(message)) hint = 'Merge consecutive same-role messages, or make sure the conversation (after an optional system message) starts with user and alternates.'
      else if (/system/i.test(message)) hint = 'This model has no system role. Fold the system prompt into the first user message.'
      else if (/tool/i.test(message)) hint = 'Check tool_call ids / tool message shape against what this template expects.'
      add('error', 'raise', `Template raised: ${message}`, `raise_exception() was hit, so apply_chat_template() throws for this conversation. ${hint}`)
    } else if (kind === 'parse') {
      add('error', 'parse', 'Template does not parse', message)
    } else {
      add('error', 'runtime', 'Template crashed while rendering', message)
    }
    return sortIssues(issues)
  }
  const text = main.text
  const msgs = base.messages
  const bos = base.specials?.bos_token

  // --- BOS
  if (bos && text.startsWith(bos)) {
    if (text.startsWith(bos + bos)) {
      add('error', 'double-bos', 'Duplicate BOS token', `The rendered prompt itself starts with ${bos}${bos}, whatever the tokenizer adds${ctx.tokenizerAddsBos ? ' (and here it adds one more)' : ''}. Models are trained on a single BOS, so this is a classic silent quality drop. Write bos_token once, and not in the first message.`)
    } else if (ctx.tokenizerAddsBos === true) {
      add('error', 'double-bos', 'Duplicate BOS token', `The template emits ${bos} and the tokenizer also prepends it on encode(). Tokenizing the rendered string with add_special_tokens=True yields two BOS tokens, a classic silent quality drop. Encode with add_special_tokens=False, or use apply_chat_template(tokenize=True).`)
    } else if (ctx.tokenizerAddsBos === null) {
      add('info', 'bos-unknown', `Template emits ${bos}`, 'If your tokenizer also prepends BOS (add_bos_token or a TemplateProcessing post-processor), encode with add_special_tokens=False to avoid a double BOS.')
    } else {
      add('ok', 'bos', 'Single BOS', `Template emits ${bos}; tokenizer does not add another.`)
    }
  } else if (bos && ctx.tokenizerAddsBos === false && !text.includes(bos)) {
    add('info', 'no-bos', 'No BOS anywhere', `Neither the template nor the tokenizer adds ${bos}. Fine for models trained that way (e.g. Qwen), surprising for Llama-style models.`)
  }

  // --- generation prompt
  const last = msgs[msgs.length - 1]
  if (base.addGenerationPrompt) {
    const noGen = run({ addGenerationPrompt: false })
    if (noGen.ok && noGen.text === text) {
      // Formats like Mistral's [/INST] have no assistant header: the prompt already hands off to the model.
      const asked = last?.role === 'assistant' ? msgs.slice(0, -1) : msgs
      const prompt = run({ messages: asked })
      const full = run({ messages: [...asked, { role: 'assistant', content: 'OK.' }], addGenerationPrompt: false })
      const before = full.ok ? full.text.slice(0, full.text.lastIndexOf('OK.')).trimEnd() : ''
      if (prompt.ok && before && prompt.text.trimEnd().endsWith(before.slice(-8))) {
        add('info', 'gen-handoff', 'No separate generation prompt', 'add_generation_prompt changes nothing here, which is fine: this format has no assistant header, so the prompt already ends where the model starts writing.')
      } else {
        add('warn', 'gen-ignored', 'add_generation_prompt is ignored', 'Rendering with and without add_generation_prompt gives identical output, but assistant turns do get a header. At inference the model never sees it, so it may continue the user turn or ramble.')
      }
    } else if (noGen.ok && text.startsWith(noGen.text)) {
      add('ok', 'gen', 'Generation prompt appended', `Adds ${JSON.stringify(text.slice(noGen.text.length))}`)
    }
    if (/[ \t]$/.test(text)) {
      add('warn', 'trailing-space', 'Generation prompt ends with a space', 'BPE/SentencePiece tokenizers merge a leading space into the next word (" Hello" is one token). A trailing space here forces an unnatural split the model rarely saw in training.')
    }
  } else if (last?.role === 'user' || last?.role === 'tool') {
    add('info', 'gen-off', 'No generation prompt', 'The conversation ends on a user turn without add_generation_prompt. For inference, turn it on, otherwise the model continues the user message instead of answering.')
  }

  // --- training/inference prefix consistency
  {
    const endsAssistant = last?.role === 'assistant'
    const prompt = endsAssistant ? run({ messages: msgs.slice(0, -1), addGenerationPrompt: true }) : run({ addGenerationPrompt: true })
    const full = endsAssistant ? run({ addGenerationPrompt: false }) : run({ messages: [...msgs, { role: 'assistant', content: 'OK.' }], addGenerationPrompt: false })
    if (prompt.ok && full.ok) {
      if (full.text.startsWith(prompt.text)) {
        add('ok', 'prefix', 'Prompt is a prefix of the training render', 'render(messages[:-1], add_generation_prompt=True) lines up with the full conversation, so loss masking and KV-cache reuse work.')
      } else {
        const at = firstDiff(prompt.text, full.text)
        const sys = msgs.find((m) => m.role === 'system')?.content
        const why = sys && prompt.text.indexOf(sys) !== full.text.indexOf(sys)
          ? 'The system prompt lands in a different turn depending on where the conversation ends (Mistral v0.3 attaches it to the last user message)'
          : 'Common with reasoning templates that strip past <think> blocks'
        add('warn', 'prefix-mismatch', 'Inference prompt diverges from training format', `The generation prompt is not a prefix of the rendered conversation (diverges at char ${at}: ${JSON.stringify(prompt.text.slice(at, at + 24))} vs ${JSON.stringify(full.text.slice(at, at + 24))}). ${why}; this breaks assistant-only loss masks and prefix caching.`)
      }
    }
  }

  // --- dropped messages
  const marked = msgs.map((m, i) => (m.content ? { ...m, content: marker(i) } : m))
  const mr = run({ messages: marked })
  if (mr.ok) {
    msgs.forEach((m, i) => {
      if (!m.content || mr.text.includes(marker(i))) return
      if (m.role === 'system') {
        add('warn', `dropped-${i}`, 'System message silently dropped', `The template never emits message #${i + 1} (system). Your instructions don't reach the model, and no error is raised.`)
      } else {
        add('warn', `dropped-${i}`, `Message #${i + 1} (${m.role}) is dropped`, `Its content never appears in the rendered prompt.${m.role === 'tool' ? ' This template has no tool role, so tool results are lost.' : ''}`)
      }
    })
  }

  // --- tools
  // A name in the prompt proves nothing by itself: it may come from a message, a schema or another call.
  // So the tool or call is renamed (`r`), and the prompt must follow. Templates that special-case or refuse
  // the new name still count when the render changed and the real name is there.
  const shown = (name: string, r: RenderResult) => (r.ok && r.text.includes(PROBE)) || ((!r.ok || r.text !== text) && text.includes(name))
  const tools = base.tools ?? []
  if (tools.length) {
    const without = run({ tools: null })
    const names = tools.map(toolName).filter(Boolean) as string[]
    if (without.ok && without.text === text) {
      add('warn', 'tools-ignored', 'Tools are ignored by this template', `Passing tools=[…] changes nothing: the model never sees ${names.map((n) => '`' + n + '`').join(', ') || 'the schemas'}. Use a tool-aware template (some repos ship a named "tool_use" template) or inject schemas into the system prompt yourself.`)
    } else {
      const missing = tools.flatMap((t, i) => {
        const name = toolName(t)
        if (!name) return [`tool #${i + 1} (has no name, so it can't be checked)`]
        const r = run({ tools: tools.map((x, j) => (j === i ? renamed(x) : x)) })
        return shown(name, r) ? [] : [name]
      })
      if (missing.length) add('warn', 'tools-partial', 'Some tools are missing from the prompt', `Not rendered: ${missing.join(', ')}.`)
      else add('ok', 'tools', `Tools rendered (${tools.length})`, 'All tool schemas appear in the prompt.')
    }
  }
  msgs.forEach((m, i) => {
    if (m.role !== 'assistant' || !Array.isArray(m.tool_calls) || !m.tool_calls.length) return
    const calls = m.tool_calls
    const dropped = calls.some((c, k) => {
      const name = toolName(c)
      if (!name) return false
      const r = run({ messages: msgs.map((x, j) => (j === i ? { ...x, tool_calls: calls.map((y, l) => (l === k ? renamed(y) : y)) } : x)) })
      return !shown(name, r)
    })
    if (dropped) {
      add('warn', `calls-${i}`, `tool_calls in message #${i + 1} are dropped`, 'The assistant’s tool calls never appear in the prompt, so the model can’t see what it called before reading the results.')
    }
  })

  // --- kwargs
  for (const k of Object.keys(base.kwargs ?? {})) {
    if (!new RegExp(`\\b${escapeRe(k)}\\b`).test(tpl)) {
      add('warn', `kwarg-${k}`, `Kwarg \`${k}\` is never read`, `This template doesn't reference \`${k}\`, so passing it does nothing (e.g. enable_thinking only exists in Qwen3-style templates).`)
    }
  }

  if (/content['"\]]*\s*\|\s*trim/.test(tpl)) {
    add('info', 'trim', 'Template trims message content', 'Leading/trailing whitespace and newlines in messages are stripped. Fine for chat, surprising for code or formatted prompts.')
  }

  return sortIssues(issues)
}

function sortIssues(issues: Issue[]) {
  return issues.sort((a, b) => rank[a.severity] - rank[b.severity])
}

function toolName(t: unknown): string | undefined {
  const o = t as { name?: string; function?: { name?: string } }
  return o?.function?.name ?? o?.name
}

/** A name no conversation contains, in characters every template and JSON writer passes through. */
const PROBE = 'tl0_renamed_probe'

/** A copy of a tool or tool call under the probe name. */
function renamed(t: unknown): unknown {
  const o = t as { name?: string; function?: { name?: string } }
  return o.function?.name !== undefined ? { ...o, function: { ...o.function, name: PROBE } } : { ...o, name: PROBE }
}

export function firstDiff(a: string, b: string) {
  let i = 0
  while (i < a.length && i < b.length && a[i] === b[i]) i++
  return i
}

const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

/** Variables a template treats as optional inputs (`x is defined`), minus the standard ones. */
export function templateKwargs(tpl: string): string[] {
  const std = new Set(['messages', 'tools', 'add_generation_prompt', 'bos_token', 'eos_token', 'documents'])
  const found = new Set<string>()
  for (const m of tpl.matchAll(/(?<![.\w])([a-z_]\w*)\s+is\s+(?:not\s+)?defined/gi)) {
    const n = m[1]
    // `set x` without a "not defined" default guard means x is a template local, not an input.
    const local = new RegExp(`set\\s+${n}\\s*=`).test(tpl) && !new RegExp(`not\\s+${n}\\s+is\\s+defined|${n}\\s+is\\s+not\\s+defined`).test(tpl)
    if (!std.has(n) && !local) found.add(n)
  }
  return [...found]
}

// ---------------------------------------------------------------- segments

export type Segment = { text: string; special: boolean }

const HEURISTIC = String.raw`<\|[^|<>\s]{1,40}\|>|<｜[^｜]{1,40}｜>|<\/?s>|\[\/?(?:INST|AVAILABLE_TOOLS|TOOL_CALLS|TOOL_RESULTS|SYSTEM_PROMPT)\]|<(?:start_of_turn|end_of_turn|bos|eos)>`

/** Split rendered text into plain and special-token runs. */
export function segment(text: string, specials: string[] = []): Segment[] {
  const known = [...new Set(specials.filter((s) => s && s.length > 1))].sort((a, b) => b.length - a.length).map(escapeRe)
  const re = new RegExp([...known, HEURISTIC].join('|'), 'g')
  const out: Segment[] = []
  let last = 0
  for (const m of text.matchAll(re)) {
    if (m.index > last) out.push({ text: text.slice(last, m.index), special: false })
    out.push({ text: m[0], special: true })
    last = m.index + m[0].length
  }
  if (last < text.length) out.push({ text: text.slice(last), special: false })
  return out
}

// ---------------------------------------------------------------- diff

export type DiffOp = { op: 'same' | 'add' | 'del'; text: string }

/** Units for diffing: special tokens and lines (newline kept with its line). */
export function diffUnits(text: string, specials: string[] = []): string[] {
  const units: string[] = []
  for (const s of segment(text, specials)) {
    if (s.special) units.push(s.text)
    else units.push(...(s.text.match(/[^\n]*\n|[^\n]+/g) ?? []))
  }
  return units
}

/** LCS diff of two unit arrays. O(n·m) memory, fine for chat-sized prompts (<~3k units). */
export function diff(a: string[], b: string[]): DiffOp[] {
  const n = a.length
  const m = b.length
  const dp = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1))
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1])
  const ops: DiffOp[] = []
  const push = (op: DiffOp['op'], text: string) => {
    const prev = ops[ops.length - 1]
    if (prev?.op === op) prev.text += text
    else ops.push({ op, text })
  }
  let i = 0
  let j = 0
  while (i < n && j < m) {
    if (a[i] === b[j]) push('same', a[i++]), j++
    else if (dp[i + 1][j] >= dp[i][j + 1]) push('del', a[i++])
    else push('add', b[j++])
  }
  while (i < n) push('del', a[i++])
  while (j < m) push('add', b[j++])
  return ops
}
