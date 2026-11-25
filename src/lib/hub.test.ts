import { test } from 'node:test'
import assert from 'node:assert/strict'
import { chooseTemplate, fetchModelTemplate, parseModelRef, refLabel, parseConfig, parseTemplates, HubError } from './hub.ts'
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

test('falls back to chat_template.json (Mistral Small 3.1 layout) and accepts /blob/ URLs', async () => {
  const { f, calls } = fakeFetch({
    'tokenizer_config.json': reply(200, { bos_token: '<s>', eos_token: '</s>', add_bos_token: true }),
    'chat_template.json': reply(200, { chat_template: '{{- bos_token }}[SYSTEM_PROMPT]{{ messages[0].content }}[/SYSTEM_PROMPT]' }),
  })
  const m = await fetchModelTemplate('https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/blob/main/chat_template.json', f)
  assert.equal(m.id, 'mistralai/Mistral-Small-3.1-24B-Instruct-2503')
  assert.equal(m.source, 'chat_template.json')
  assert.equal(m.addBosToken, true)
  assert.ok(calls.some((u) => u.endsWith('/chat_template.jinja')))
})

test('chat_template.jinja takes priority over tokenizer_config, as in transformers', async () => {
  const { f } = fakeFetch({
    'tokenizer_config.json': reply(200, { chat_template: 'old', eos_token: '</s>' }),
    'chat_template.jinja': reply(200, 'new'),
  })
  const m = await fetchModelTemplate('org/both', f)
  assert.equal(m.source, 'chat_template.jinja')
  assert.deepEqual(m.templates, [{ name: 'default', template: 'new' }])
  assert.equal(m.specials.eos_token, '</s>')
})

test('model references: ids, URLs and revisions', () => {
  const ref = (raw: string) => {
    const r = parseModelRef(raw)
    return r && refLabel(r)
  }
  for (const raw of ['org/m', ' org/m/ ', 'https://huggingface.co/org/m', 'http://huggingface.co/org/m/', 'https://huggingface.co/org/m?library=transformers', 'https://huggingface.co/org/m#usage', 'https://huggingface.co/org/m/tree/main', 'https://huggingface.co/org/m/blob/main/a/b.json', 'https://huggingface.co/org/m/tree/', 'org/m/tree/main'])
    assert.equal(ref(raw), 'org/m', raw)
  assert.equal(ref('gpt2'), 'gpt2')
  assert.equal(ref('https://huggingface.co/org/m/tree/v1.0'), 'org/m@v1.0')
  assert.equal(ref('https://huggingface.co/org/m/resolve/3f2a1bc/sub/chat_template.jinja?download=true'), 'org/m@3f2a1bc')
  assert.equal(ref('https://huggingface.co/org/m/tree/refs%2Fpr%2F3'), 'org/m@refs/pr/3')
  assert.equal(ref('https://huggingface.co/org/m/blob/refs/pr/3/tokenizer_config.json'), 'org/m@refs/pr/3')
  for (const raw of ['', 'not a model id', 'a/b/c', 'https://huggingface.co/datasets/org/d', 'https://huggingface.co/org/m/tree/%', 'https://huggingface.co/org/m/tree/..', 'https://example.com/org/m'])
    assert.equal(ref(raw), null, raw)
})

test('files are fetched at the revision, and a missing revision is named', async () => {
  const { f, calls } = fakeFetch({ 'resolve/refs%2Fpr%2F3/chat_template.jinja': reply(200, 'PR') })
  const m = await fetchModelTemplate('https://huggingface.co/org/m/tree/refs%2Fpr%2F3', f)
  assert.equal(m.templates[0].template, 'PR')
  assert.equal(m.revision, 'refs/pr/3')
  assert.ok(calls.every((u) => u.includes('/org/m/resolve/refs%2Fpr%2F3/')))
  const gone = fakeFetch({ 'tokenizer_config.json': reply(404, '', { 'x-error-code': 'RevisionNotFound' }) })
  await assert.rejects(fetchModelTemplate('https://huggingface.co/org/m/tree/v9', gone.f), (e: HubError) => e.kind === 'not-found' && /no revision called v9/.test(e.message))
})

test('template choice follows get_chat_template in transformers', () => {
  const t = (...names: string[]) => parseTemplates(names.map((name) => ({ name, template: name.toUpperCase() })))
  const pick = (names: string[], hasTools: boolean, name?: string) => {
    const c = chooseTemplate(t(...names), hasTools, name)
    return c && (c.name ?? c.error)
  }
  assert.equal(pick(['tool_use', 'default'], false), 'default')
  assert.equal(pick(['default', 'tool_use'], true), 'tool_use')
  assert.equal(pick(['default', 'rag'], true), 'default')
  // An explicit pick wins; one the model does not have is no pick.
  assert.equal(pick(['default', 'tool_use'], true, 'default'), 'default')
  assert.equal(pick(['rag', 'tool_use'], false, 'rag'), 'rag')
  assert.equal(pick(['default', 'tool_use'], true, 'gone'), 'tool_use')
  assert.match(pick(['tool_use', 'rag'], false)!, /no default specified!.*\['rag', 'tool_use'\]/)
  assert.match(pick(['rag'], true)!, /no default specified/)
  assert.equal(pick([], true), null)
  // The list becomes a dict: the last entry of a name wins.
  const dup = chooseTemplate(parseTemplates([{ name: 'default', template: 'A' }, { name: 'default', template: 'B' }]), false)
  assert.ok(dup && dup.name !== null && dup.template === 'B')
})
