import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { render, lint, segment, diff, diffUnits, templateKwargs, CHATML_FALLBACK, type ChatMessage, type RenderInput } from './template.ts'

type Fx = { chat_template: string; bos_token: string | null; eos_token: string; add_bos_token: boolean | null }
const fx: Record<string, Fx> = JSON.parse(readFileSync(new URL('./real-templates.fixture.json', import.meta.url), 'utf8'))

const chat: ChatMessage[] = [
  { role: 'system', content: 'You are terse.' },
  { role: 'user', content: 'Hi' },
]
const weather = [{ type: 'function', function: { name: 'get_weather', description: 'Weather', parameters: { type: 'object', properties: { city: { type: 'string' } } } } }]

function input(name: string, over: Partial<RenderInput> = {}): RenderInput {
  const f = fx[name]
  return {
    template: f.chat_template,
    messages: chat,
    addGenerationPrompt: true,
    specials: { bos_token: f.bos_token ?? undefined, eos_token: f.eos_token },
    ...over,
  }
}
const lintFor = (name: string, over: Partial<RenderInput> = {}, adds: boolean | null = fx[name].add_bos_token) => {
  const { template, ...rest } = input(name, over)
  return lint({ template, input: rest, tokenizerAddsBos: adds })
}
const ids = (issues: { id: string }[]) => issues.map((i) => i.id)

test('renders Qwen2.5 ChatML with generation prompt', () => {
  const r = render(input('qwen25'))
  assert.ok(r.ok)
  assert.equal(r.text, '<|im_start|>system\nYou are terse.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n')
})

test('raise_exception becomes a structured raise error (Gemma 2 rejects system)', () => {
  const r = render(input('gemma2'))
  assert.equal(r.ok, false)
  assert.ok(!r.ok && r.error.kind === 'raise' && r.error.message === 'System role not supported')
  const issues = lintFor('gemma2')
  assert.equal(issues[0].id, 'raise')
  assert.match(issues[0].detail, /Fold the system prompt/)
})

test('role alternation violation is reported with a hint (Mistral v0.3)', () => {
  const issues = lintFor('mistral03', { messages: [{ role: 'user', content: 'a' }, { role: 'user', content: 'b' }] })
  assert.equal(issues[0].id, 'raise')
  assert.match(issues[0].title, /alternate/)
  assert.match(issues[0].detail, /Merge consecutive/)
})

test('parse errors are reported, not thrown', () => {
  const r = render({ template: '{% for x in %}', messages: chat, addGenerationPrompt: true })
  assert.ok(!r.ok && r.error.kind === 'parse')
})

test('duplicate BOS: Llama 3.1 template + add_bos_token tokenizer', () => {
  const issues = lintFor('llama31')
  assert.ok(ids(issues).includes('double-bos'))
  assert.equal(issues.find((i) => i.id === 'double-bos')!.severity, 'error')
  assert.ok(!ids(lintFor('llama31', {}, false)).includes('double-bos'))
  assert.ok(ids(lintFor('llama31', {}, null)).includes('bos-unknown'))
})

test('system message silently dropped is detected', () => {
  const tpl = "{% for m in messages %}{% if m.role != 'system' %}<|{{ m.role }}|>{{ m.content }}{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
  const issues = lint({ template: tpl, input: { messages: chat, addGenerationPrompt: true }, tokenizerAddsBos: false })
  const d = issues.find((i) => i.id === 'dropped-0')
  assert.ok(d)
  assert.match(d.title, /System message silently dropped/)
})

test('tools ignored by template (Phi-3.5) vs rendered (Qwen2.5)', () => {
  assert.ok(ids(lintFor('phi35', { tools: weather })).includes('tools-ignored'))
  const q = lintFor('qwen25', { tools: weather })
  assert.ok(ids(q).includes('tools'))
  assert.ok(!ids(q).includes('tools-ignored'))
})

test('tool role dropped when template has no tool branch (Phi-3.5)', () => {
  const msgs: ChatMessage[] = [
    { role: 'user', content: 'Weather in Paris?' },
    { role: 'assistant', content: '', tool_calls: [{ type: 'function', function: { name: 'get_weather', arguments: { city: 'Paris' } } }] },
    { role: 'tool', content: '{"temp": 21}' },
  ]
  const issues = lintFor('phi35', { messages: msgs })
  assert.ok(issues.some((i) => i.id === 'dropped-2' && /tool results are lost/.test(i.detail)))
  assert.ok(ids(issues).includes('calls-1'))
})

test('add_generation_prompt ignored is flagged', () => {
  const tpl = '{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}'
  const issues = lint({ template: tpl, input: { messages: chat, addGenerationPrompt: true }, tokenizerAddsBos: false })
  assert.ok(ids(issues).includes('gen-ignored'))
})

test('Mistral [/INST] hand-off is not a false "generation prompt ignored"', () => {
  const issues = ids(lintFor('mistral03'))
  assert.ok(issues.includes('gen-handoff'))
  assert.ok(!issues.includes('gen-ignored'))
})

test('Mistral v0.3 drops the system prompt unless the conversation ends on user', () => {
  const msgs: ChatMessage[] = [...chat, { role: 'assistant', content: 'Yo' }]
  assert.ok(ids(lintFor('mistral03', { messages: msgs, addGenerationPrompt: false })).includes('dropped-0'))
  assert.ok(!ids(lintFor('mistral03')).includes('dropped-0'))
})

test('trailing space after generation prompt is flagged', () => {
  const tpl = "{% for m in messages %}[{{ m.role }}] {{ m.content }}\n{% endfor %}{% if add_generation_prompt %}[assistant] {% endif %}"
  assert.ok(ids(lint({ template: tpl, input: { messages: chat, addGenerationPrompt: true }, tokenizerAddsBos: false })).includes('trailing-space'))
})

test('prefix consistency: ChatML ok, think-stripping template mismatches', () => {
  assert.ok(ids(lintFor('qwen25')).includes('prefix'))
  // A template whose generation prompt opens <think> but whose rendered assistant turns do not.
  const tpl = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}<|end|>{% endfor %}{% if add_generation_prompt %}<|assistant|><think>\n{% endif %}"
  const issues = lint({ template: tpl, input: { messages: chat, addGenerationPrompt: true }, tokenizerAddsBos: false })
  const p = issues.find((i) => i.id === 'prefix-mismatch')
  assert.ok(p)
  assert.match(p.detail, /<think>/)
})

test('unused kwarg is flagged; Qwen3 enable_thinking is not', () => {
  assert.ok(ids(lintFor('qwen25', { kwargs: { enable_thinking: false } })).includes('kwarg-enable_thinking'))
  assert.ok(!ids(lintFor('qwen3', { kwargs: { enable_thinking: false } })).includes('kwarg-enable_thinking'))
  const r = render(input('qwen3', { kwargs: { enable_thinking: false } }))
  assert.ok(r.ok && r.text.endsWith('<|im_start|>assistant\n<think>\n\n</think>\n\n'))
})

test('missing template falls back to ChatML and is an error', () => {
  const issues = lint({ template: null, input: { messages: chat, addGenerationPrompt: true }, tokenizerAddsBos: null })
  assert.equal(issues[0].id, 'missing-template')
  assert.ok(render({ template: CHATML_FALLBACK, messages: chat, addGenerationPrompt: true }).ok)
})

test('templateKwargs finds optional inputs', () => {
  assert.deepEqual(templateKwargs(fx.qwen3.chat_template), ['enable_thinking'])
  assert.ok(templateKwargs(fx.llama31.chat_template).includes('date_string'))
  assert.deepEqual(templateKwargs(fx.mistral03.chat_template), [])
})

test('segment splits known and heuristic special tokens', () => {
  const segs = segment('<|im_start|>user\nHi<|im_end|><s>[INST] x [/INST]', ['<|im_start|>'])
  assert.deepEqual(segs.filter((s) => s.special).map((s) => s.text), ['<|im_start|>', '<|im_end|>', '<s>', '[INST]', '[/INST]'])
  assert.equal(segs.map((s) => s.text).join(''), '<|im_start|>user\nHi<|im_end|><s>[INST] x [/INST]')
})

test('diff on units keeps both sides reconstructable', () => {
  const a = '<|im_start|>user\nHi<|im_end|>\n'
  const b = '<s>[INST] Hi [/INST]'
  const ops = diff(diffUnits(a), diffUnits(b))
  assert.equal(ops.filter((o) => o.op !== 'add').map((o) => o.text).join(''), a)
  assert.equal(ops.filter((o) => o.op !== 'del').map((o) => o.text).join(''), b)
  assert.deepEqual(diff(['x', 'y'], ['x', 'y']), [{ op: 'same', text: 'xy' }])
})
