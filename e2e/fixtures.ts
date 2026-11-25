import { test as base, expect, type Page } from '@playwright/test'
import { readFileSync } from 'node:fs'

/** The only other hosts the page asks for: web fonts in index.html. They are answered with an empty stylesheet. */
const FONT_HOSTS = new Set(['fonts.googleapis.com', 'fonts.gstatic.com'])
const HUB = 'https://huggingface.co'

/** One answer of the stubbed Hub. */
export interface Reply { status?: number; body?: unknown; headers?: Record<string, string> }
type Route = Reply | (() => Reply)

const real: Record<string, { chat_template: string; bos_token: string | null; eos_token: string; add_bos_token: boolean | null }> =
  JSON.parse(readFileSync(new URL('../src/lib/real-templates.fixture.json', import.meta.url), 'utf8'))

const realConfig = (name: string) => {
  const { chat_template, bos_token, eos_token, add_bos_token } = real[name]
  return { body: { chat_template, bos_token, eos_token, ...(add_bos_token === null ? {} : { add_bos_token }) } }
}

/**
 * A tokenizer small enough to work out on paper: a word-level vocabulary split on single spaces.
 * The vocabulary is the whole model, so every id below can be read off this table.
 */
export const VOCAB = { '<unk>': 0, '<s>': 1, '</s>': 2, alpha: 3, beta: 4, gamma: 5, hi: 6 }
export const wordTokenizer = {
  version: '1.0',
  truncation: null,
  padding: null,
  added_tokens: [0, 1, 2].map((id) => ({ id, content: ['<unk>', '<s>', '</s>'][id], single_word: false, lstrip: false, rstrip: false, normalized: false, special: true })),
  normalizer: null,
  pre_tokenizer: { type: 'Whitespace' },
  post_processor: null,
  decoder: null,
  model: { type: 'WordLevel', vocab: VOCAB, unk_token: '<unk>' },
}
/** Same vocabulary, but encode() prepends <s> through a TemplateProcessing post-processor. */
export const bosTokenizer = {
  ...wordTokenizer,
  post_processor: {
    type: 'TemplateProcessing',
    single: [{ SpecialToken: { id: '<s>', type_id: 0 } }, { Sequence: { id: 'A', type_id: 0 } }],
    pair: [{ Sequence: { id: 'A', type_id: 0 } }, { Sequence: { id: 'B', type_id: 1 } }],
    special_tokens: { '<s>': { id: '<s>', ids: [1], tokens: ['<s>'] } },
  },
}

const PLAIN = '{{ bos_token }}{% for m in messages %}{{ m.content }}{{ eos_token }}{% endfor %}'
const tinyConfig = { tokenizer_class: 'PreTrainedTokenizerFast', bos_token: '<s>', eos_token: '</s>', unk_token: '<unk>', chat_template: PLAIN }

/** Files the stubbed Hub serves, by "<id>/<file>". */
const HUB_FILES: Record<string, Route> = {
  'Qwen/Qwen2.5-7B-Instruct/tokenizer_config.json': realConfig('qwen25'),
  'Qwen/Qwen3-8B/tokenizer_config.json': realConfig('qwen3'),
  'mistralai/Mistral-7B-Instruct-v0.3/tokenizer_config.json': realConfig('mistral03'),
  'microsoft/Phi-3.5-mini-instruct/tokenizer_config.json': realConfig('phi35'),
  'unsloth/Llama-3.1-8B-Instruct/tokenizer_config.json': realConfig('llama31'),
  'unsloth/gemma-2-2b-it/tokenizer_config.json': realConfig('gemma2'),
  'test/jinja-wins/tokenizer_config.json': { body: { chat_template: 'OLD', eos_token: '</s>' } },
  'test/jinja-wins/chat_template.jinja': { body: 'JINJA:{{ messages[0].content }}' },
  'test/processor/tokenizer_config.json': { body: { bos_token: '<s>' } },
  'test/processor/chat_template.json': { body: { chat_template: 'PROC:{{ messages[0].content }}' } },
  'test/named/tokenizer_config.json': { body: { chat_template: [{ name: 'tool_use', template: 'TOOLS:{{ tools | length }}' }, { name: 'default', template: 'DEFAULT' }] } },
  'test/base-model/tokenizer_config.json': { body: { bos_token: '<s>' } },
  'test/tiny/tokenizer_config.json': { body: tinyConfig },
  'test/tiny/tokenizer.json': { body: wordTokenizer },
  'test/tiny-bos/tokenizer_config.json': { body: tinyConfig },
  'test/tiny-bos/tokenizer.json': { body: bosTokenizer },
  'test/tiny-long/tokenizer_config.json': { body: { ...tinyConfig, chat_template: '{{ bos_token }} gamma gamma {% for m in messages %}{{ m.content }}{{ eos_token }}{% endfor %}' } },
  'test/tiny-long/tokenizer.json': { body: wordTokenizer },
  'test/empty-config/tokenizer_config.json': { body: '' },
  'test/truncated-config/tokenizer_config.json': { body: '{"chat_template": "{{ messages[0].con' },
  'test/html-config/tokenizer_config.json': { body: '<!doctype html><title>Sign in</title>' },
  'test/broken/tokenizer_config.json': { status: 503, body: 'down' },
  'test/truncated-processor/tokenizer_config.json': { body: { bos_token: '<s>' } },
  'test/truncated-processor/chat_template.json': { body: '{"chat_template": "PROC:{{ mess' },
  'meta-llama/Llama-3.1-8B-Instruct/tokenizer_config.json': { status: 401, headers: { 'x-error-code': 'GatedRepo' } },
}

/** Every request the page made to the stubbed Hub, as a path. */
type Hub = { files: Record<string, Route>; seen: string[] }

export const test = base.extend<{ hub: Hub; external: string[] }>({
  external: async ({}, use) => { await use([]) },
  hub: async ({}, use) => { await use({ files: { ...HUB_FILES }, seen: [] }) },

  // Only the app is reachable. The Hub is answered from the table above, any other host fails the test,
  // and so does any uncaught page error.
  context: async ({ context, baseURL, hub, external }, use) => {
    const errors: string[] = []
    const origin = new URL(baseURL!).origin
    await context.route('**/*', (route) => {
      const url = new URL(route.request().url())
      if (url.origin === origin) return route.continue()
      if (FONT_HOSTS.has(url.host)) return route.fulfill({ status: 200, contentType: 'text/css', body: '' })
      if (url.origin !== HUB) {
        external.push(url.host)
        return route.abort('blockedbyclient')
      }
      hub.seen.push(url.pathname)
      const r = hubReply(hub.files, url.pathname)
      const cors = { 'access-control-allow-origin': '*', 'access-control-expose-headers': 'x-error-code' }
      return route.fulfill({ status: r.status ?? 200, headers: { 'content-type': 'application/json', ...cors, ...r.headers }, body: typeof r.body === 'string' ? r.body : r.body === undefined ? '' : JSON.stringify(r.body) })
    })
    context.on('weberror', (e) => errors.push(String(e.error())))
    // Replaced so a test can read what the page copied without browser permissions.
    await context.addInitScript(() => {
      const w = window as unknown as { __copied: string[] }
      w.__copied = []
      Object.defineProperty(navigator, 'clipboard', { value: { writeText: async (t: string) => { w.__copied.push(t) } }, configurable: true })
    })
    await use(context)
    expect(errors, 'uncaught page errors').toEqual([])
    expect(external, 'requests to hosts other than the app and the Hub').toEqual([])
  },
})

/** Answers as huggingface.co does: 404 EntryNotFound for a missing file, 401 for an unknown repo. */
function hubReply(files: Record<string, Route>, path: string): Reply {
  const m = path.match(/^\/([^/]+\/[^/]+)\/resolve\/([^/]+)\/(.+)$/)
  if (!m) return { status: 401, headers: { 'x-error-code': 'RepoNotFound' } }
  const [, id, rev, file] = m
  const hit = files[rev === 'main' ? `${id}/${file}` : `${id}@${rev}/${file}`]
  if (hit) return typeof hit === 'function' ? hit() : hit
  // A gated repo hides every file.
  const gated = files[`${id}/tokenizer_config.json`] as Reply | undefined
  if (gated?.headers?.['x-error-code'] === 'GatedRepo') return gated
  const known = Object.keys(files).some((k) => k.startsWith(rev === 'main' ? `${id}/` : `${id}@${rev}/`))
  return known ? { status: 404, headers: { 'x-error-code': 'EntryNotFound' } } : { status: 401, headers: { 'x-error-code': 'RepoNotFound' } }
}

export { expect }

export const copied = (page: Page) => page.evaluate(() => (window as unknown as { __copied: string[] }).__copied.at(-1) ?? '')

// ---------- Driving the lab ----------

export interface Setup {
  model: string
  override: string | null
  tplName: string
  pasteBos: string
  pasteEos: string
  compare: string
  msgs: { role: string; content: string; toolCalls?: string; toolCallId?: string }[]
  toolsOn: boolean
  tools: string
  agp: boolean
  kwargs: string
  tokenize: boolean
}

/** Opens the lab. */
export async function openLab(page: Page, hash = '') {
  await page.goto('/app' + hash)
  await expect(page.getByRole('heading', { name: 'Linter' })).toBeVisible()
}

export const tab = (page: Page, name: 'Rendered' | 'Tokens' | 'Compare models' | 'Template') =>
  page.getByRole('tablist', { name: 'Output views' }).getByRole('tab', { name: new RegExp(`^${name}`) }).click()

/** Switches to paste mode and types a template. */
export async function paste(page: Page, template: string, bos = '', eos = '') {
  await page.getByRole('tab', { name: 'Paste a template' }).click()
  await page.getByLabel('Chat template source').fill(template)
  await page.getByLabel('bos_token').fill(bos)
  await page.getByLabel('eos_token').fill(eos)
}

/** Replaces the conversation through the editor. */
export async function conversation(page: Page, msgs: { role: string; content: string; toolCalls?: string }[]) {
  const del = page.getByRole('button', { name: /^Delete message/ })
  while (await del.count()) await del.first().click()
  for (const [i, m] of msgs.entries()) {
    await page.locator('.add-row').getByRole('button', { name: `+ ${m.role}` }).click()
    await page.getByLabel(`Message ${i + 1} content`).fill(m.content)
    if (m.toolCalls !== undefined) {
      await page.locator('.msg').nth(i).getByRole('button', { name: '+ tool_calls' }).click()
      await page.getByLabel(`Message ${i + 1} tool_calls JSON`).fill(m.toolCalls)
    }
  }
}

export async function loadModel(page: Page, id: string) {
  await page.getByLabel('Model id').fill(id)
  await page.getByRole('button', { name: 'Load', exact: true }).click()
}

/** The rendered prompt, exactly as "Copy" gives it. */
export async function prompt(page: Page): Promise<string> {
  await tab(page, 'Rendered')
  await page.getByRole('button', { name: 'Copy', exact: true }).click()
  return copied(page)
}

export const issue = (page: Page, title: string | RegExp) => page.locator('.card.lint .issue').filter({ has: page.locator('strong', { hasText: title }) })

export const setSwitch = (page: Page, name: 'add_generation_prompt' | 'tools', on: boolean) =>
  page.locator('label.switch').filter({ has: page.locator('code', { hasText: new RegExp(`^${name}$`) }) }).locator('input').setChecked(on, { force: true })

export const stored = (page: Page) => page.evaluate(() => JSON.parse(localStorage.getItem('tl:state') ?? 'null')) as Promise<Setup | null>

/** A share link's payload, built the way the app documents it: JSON, raw deflate, base64url. */
export async function shareHash(state: unknown): Promise<string> {
  const { deflateRawSync } = await import('node:zlib')
  return '#s=' + deflateRawSync(Buffer.from(JSON.stringify(state))).toString('base64url')
}
