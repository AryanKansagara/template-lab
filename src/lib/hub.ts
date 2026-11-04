import type { SpecialTokens } from './template.ts'

export const HUB = 'https://huggingface.co'

export interface NamedTemplate {
  name: string
  template: string
}

export interface ModelTemplate {
  id: string
  /** null when the repo ships no chat template at all */
  templates: NamedTemplate[]
  source: 'tokenizer_config.json' | 'chat_template.jinja' | 'chat_template.json' | 'none'
  specials: SpecialTokens
  /** tokenizer_config.add_bos_token; null if unset */
  addBosToken: boolean | null
  /** every token flagged special in added_tokens_decoder */
  specialTokens: string[]
}

export type HubErrorKind = 'gated' | 'not-found' | 'invalid' | 'network'

export class HubError extends Error {
  kind: HubErrorKind
  suggestions: string[]
  constructor(kind: HubErrorKind, message: string, suggestions: string[] = []) {
    super(message)
    this.kind = kind
    this.suggestions = suggestions
  }
}

/** Ungated models with good chat templates. */
export const DEFAULT_MODELS = [
  'Qwen/Qwen2.5-7B-Instruct',
  'Qwen/Qwen3-8B',
  'mistralai/Mistral-7B-Instruct-v0.3',
  'HuggingFaceTB/SmolLM2-1.7B-Instruct',
  'microsoft/Phi-3.5-mini-instruct',
  'unsloth/Llama-3.1-8B-Instruct',
  'unsloth/gemma-2-2b-it',
  'openai/gpt-oss-20b',
  'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
]

/** Well-known ungated re-uploads of gated repos. */
const MIRRORS: Record<string, string> = {
  'meta-llama/llama-3.1-8b-instruct': 'unsloth/Llama-3.1-8B-Instruct',
  'meta-llama/meta-llama-3.1-8b-instruct': 'unsloth/Llama-3.1-8B-Instruct',
  'meta-llama/llama-3.2-1b-instruct': 'unsloth/Llama-3.2-1B-Instruct',
  'meta-llama/llama-3.2-3b-instruct': 'unsloth/Llama-3.2-3B-Instruct',
  'meta-llama/meta-llama-3-8b-instruct': 'NousResearch/Meta-Llama-3-8B-Instruct',
  'google/gemma-2-2b-it': 'unsloth/gemma-2-2b-it',
  'google/gemma-2-9b-it': 'unsloth/gemma-2-9b-it',
  'google/gemma-3-1b-it': 'unsloth/gemma-3-1b-it',
  'google/gemma-3-4b-it': 'unsloth/gemma-3-4b-it',
}

export const ID_RE = /^[\w.-]+(\/[\w.-]+)?$/

type Fetch = typeof fetch

const resolve = (id: string, file: string) => `${HUB}/${id}/resolve/main/${file}`

const tokenContent = (v: unknown): string | undefined =>
  typeof v === 'string' ? v : v && typeof v === 'object' && typeof (v as { content?: unknown }).content === 'string' ? (v as { content: string }).content : undefined

export function mirrorFor(id: string): string | undefined {
  return MIRRORS[id.toLowerCase()]
}

/** Known mirror, or an unsloth re-upload if the Hub confirms one is public. Best effort. */
async function findMirror(id: string, f: Fetch): Promise<string[]> {
  const known = mirrorFor(id)
  if (known) return [known]
  const guess = `unsloth/${id.split('/').pop()}`
  try {
    const res = await f(`${HUB}/api/models/${guess}`)
    if (res.ok && !(await res.json()).gated) return [guess]
  } catch {
    // no suggestion is fine; the UI still offers the default models
  }
  return []
}

async function get(id: string, file: string, f: Fetch): Promise<Response | null> {
  let res: Response
  try {
    res = await f(resolve(id, file))
  } catch {
    throw new HubError('network', `Couldn't reach huggingface.co. Check your connection.`)
  }
  if (res.ok) return res
  if (res.headers.get('x-error-code') === 'GatedRepo') {
    throw new HubError('gated', `${id} is gated. Its files need a logged-in, approved account, so a browser can't fetch them anonymously.`, await findMirror(id, f))
  }
  if (res.status === 404 && res.headers.get('x-error-code') === 'EntryNotFound') return null
  if (res.status === 401 || res.status === 404) {
    throw new HubError('not-found', `No public model called ${id}. Check the spelling, or it may be private.`)
  }
  throw new HubError('network', `huggingface.co answered ${res.status} for ${file}.`)
}

/** Parse the chat_template field: a string, or a list of {name, template}. */
export function parseTemplates(v: unknown): NamedTemplate[] {
  if (typeof v === 'string') return v.trim() ? [{ name: 'default', template: v }] : []
  if (Array.isArray(v)) {
    const out = v.filter((t) => t && typeof t.template === 'string').map((t) => ({ name: String(t.name ?? 'default'), template: t.template as string }))
    return out.sort((a, b) => Number(b.name === 'default') - Number(a.name === 'default'))
  }
  return []
}

export function parseConfig(id: string, cfg: Record<string, unknown>): ModelTemplate {
  const specials: SpecialTokens = {}
  for (const k of ['bos_token', 'eos_token', 'unk_token', 'pad_token'] as const) {
    const t = tokenContent(cfg[k])
    if (t) specials[k] = t
  }
  const decoder = (cfg.added_tokens_decoder ?? {}) as Record<string, { content?: string; special?: boolean }>
  const specialTokens = Object.values(decoder)
    .filter((t) => t?.special && t.content)
    .map((t) => t.content!)
  for (const v of Object.values(specials)) if (v && !specialTokens.includes(v)) specialTokens.push(v)
  const templates = parseTemplates(cfg.chat_template)
  return {
    id,
    templates,
    source: templates.length ? 'tokenizer_config.json' : 'none',
    specials,
    addBosToken: typeof cfg.add_bos_token === 'boolean' ? cfg.add_bos_token : null,
    specialTokens,
  }
}

export async function fetchModelTemplate(rawId: string, f: Fetch = fetch): Promise<ModelTemplate> {
  const id = rawId.trim().replace(/^https?:\/\/huggingface\.co\//, '').replace(/\/(tree|blob|resolve)\/.*$/, '').replace(/\/$/, '')
  if (!ID_RE.test(id)) throw new HubError('invalid', 'Model ids look like org/name, e.g. Qwen/Qwen2.5-7B-Instruct.')

  const cfgRes = await get(id, 'tokenizer_config.json', f)
  let cfg: Record<string, unknown> = {}
  if (cfgRes) {
    try {
      cfg = await cfgRes.json()
    } catch {
      throw new HubError('invalid', 'tokenizer_config.json is not valid JSON.')
    }
  }
  const model = parseConfig(id, cfg)
  if (model.templates.length) return model

  // Newer repos keep the template in its own file.
  const jinja = await get(id, 'chat_template.jinja', f)
  if (jinja) return { ...model, templates: [{ name: 'default', template: await jinja.text() }], source: 'chat_template.jinja' }
  const json = await get(id, 'chat_template.json', f)
  if (json) {
    const templates = parseTemplates((await json.json().catch(() => ({})))?.chat_template)
    if (templates.length) return { ...model, templates, source: 'chat_template.json' }
  }
  if (!cfgRes) throw new HubError('not-found', `${id} has no tokenizer_config.json or chat template files.`)
  return model
}
