import { test } from 'node:test'
import assert from 'node:assert/strict'
import { fetchModelTemplate, parseConfig, parseTemplates, HubError } from './hub.ts'
import { encodeState, decodeState } from './share.ts'

const reply = (status: number, body: unknown = '', headers: Record<string, string> = {}) =>
  new Response(typeof body === 'string' ? body : JSON.stringify(body), { status, headers })

function fakeFetch(routes: Record<string, Response>) {
  const calls: string[] = []
  const f = (async (url: string) => {
    calls.push(url)
    const hit = Object.entries(routes).find(([k]) => url.endsWith(k))
    return hit ? hit[1].clone() : reply(404, '', { 'x-error-code': 'EntryNotFound' })
  }) as typeof fetch
  return { f, calls }
}

test('parses tokenizer_config with AddedToken objects and specials', () => {
  const m = parseConfig('x/y', {
    chat_template: '{{ bos_token }}',
    bos_token: { __type: 'AddedToken', content: '<｜begin▁of▁sentence｜>' },
    eos_token: '</s>',
    add_bos_token: true,
    added_tokens_decoder: { '0': { content: '<unk>', special: true }, '5': { content: 'word', special: false } },
  })
  assert.equal(m.specials.bos_token, '<｜begin▁of▁sentence｜>')
  assert.equal(m.addBosToken, true)
  assert.deepEqual(m.specialTokens.sort(), ['</s>', '<unk>', '<｜begin▁of▁sentence｜>'].sort())
})

test('named template list puts default first', () => {
  const t = parseTemplates([{ name: 'tool_use', template: 'B' }, { name: 'default', template: 'A' }])
  assert.deepEqual(t.map((x) => x.name), ['default', 'tool_use'])
})

test('falls back to chat_template.jinja for newer repos', async () => {
  const { f } = fakeFetch({
    'tokenizer_config.json': reply(200, { eos_token: '<|return|>' }),
    'chat_template.jinja': reply(200, '{{ messages[0].content }}'),
  })
  const m = await fetchModelTemplate('https://huggingface.co/openai/gpt-oss-20b/tree/main', f)
  assert.equal(m.id, 'openai/gpt-oss-20b')
  assert.equal(m.source, 'chat_template.jinja')
  assert.equal(m.templates[0].template, '{{ messages[0].content }}')
})

test('gated repo becomes a gated HubError with an ungated mirror', async () => {
  const { f } = fakeFetch({ 'tokenizer_config.json': reply(401, '', { 'x-error-code': 'GatedRepo' }) })
  await assert.rejects(fetchModelTemplate('meta-llama/Llama-3.1-8B-Instruct', f), (e: HubError) => e.kind === 'gated' && e.suggestions[0] === 'unsloth/Llama-3.1-8B-Instruct')
})

test('gated repo without a known mirror only suggests a verified copy', async () => {
  const gated = reply(401, '', { 'x-error-code': 'GatedRepo' })
  const hit = fakeFetch({ 'tokenizer_config.json': gated, 'unsloth/new-model': reply(200, { gated: false }) })
  await assert.rejects(fetchModelTemplate('acme/new-model', hit.f), (e: HubError) => e.suggestions[0] === 'unsloth/new-model')
  const miss = fakeFetch({ 'tokenizer_config.json': gated, 'unsloth/new-model': reply(401) })
  await assert.rejects(fetchModelTemplate('acme/new-model', miss.f), (e: HubError) => e.kind === 'gated' && e.suggestions.length === 0)
})

test('unknown repo and bad ids are reported', async () => {
  const { f } = fakeFetch({ 'tokenizer_config.json': reply(401) })
  await assert.rejects(fetchModelTemplate('nope/nope', f), (e: HubError) => e.kind === 'not-found')
  await assert.rejects(fetchModelTemplate('not a model id', f), (e: HubError) => e.kind === 'invalid')
})

test('repo without any template returns source none', async () => {
  const { f } = fakeFetch({ 'tokenizer_config.json': reply(200, { bos_token: '<s>' }) })
  const m = await fetchModelTemplate('org/base-model', f)
  assert.equal(m.source, 'none')
  assert.equal(m.templates.length, 0)
})

test('share state round-trips through the URL encoding', async () => {
  const state = { a: 'Qwen/Qwen3-8B', msgs: [{ role: 'user', content: 'héllo <|im_end|> 🤗'.repeat(50) }] }
  const s = await encodeState(state)
  assert.match(s, /^[\w-]+$/)
  assert.ok(s.length < JSON.stringify(state).length)
  assert.deepEqual(await decodeState(s), state)
  assert.equal(await decodeState('garbage!!'), null)
})
