import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { render, lint, pyRepr, segment, diff, diffUnits, templateKwargs, CHATML_FALLBACK, type ChatMessage, type RenderInput } from './template.ts'
import { parsePython } from './json.ts'

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

test('SmolLM3 renders tools with `| string` exactly as Python repr()', () => {
  const tools = [{ type: 'function', function: { name: 'get_weather', description: "Get today's weather", parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'], strict: true, x: null } } }]
  const r = render(input('smollm3', { messages: [{ role: 'user', content: 'hi' }], tools }))
  assert.ok(r.ok)
  // Expected line produced by CPython 3 str(dict) on the same tool.
  assert.ok(r.text.includes(`{'type': 'function', 'function': {'name': 'get_weather', 'description': "Get today's weather", 'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city'], 'strict': True, 'x': None}}}`))
  assert.ok(ids(lintFor('smollm3', { tools })).includes('tools'))
})

test('pyRepr matches Python for strings, nesting and floats', () => {
  assert.equal(pyRepr(new Map<string, unknown>([['a', 'it\'s "x"'], ['b', [1, false]]])), `{'a': 'it\\'s "x"', 'b': [1, False]}`)
  assert.equal(pyRepr({ type: 'FloatValue', value: 18 }), '18.0')
  assert.deepEqual(render({ template: '{{ x | string }}|{{ (y | string) }}', messages: [], addGenerationPrompt: false, kwargs: { x: 'plain', y: [1, 'a'] } }), { ok: true, text: "plain|[1, 'a']" })
})

test('Mistral v0.3 rejects tool call ids that are not 9 characters', () => {
  const calls = (id: string): ChatMessage[] => [
    { role: 'user', content: 'Weather?' },
    { role: 'assistant', content: '', tool_calls: [{ id, type: 'function', function: { name: 'get_weather', arguments: { city: 'Paris' } } }] },
    { role: 'tool', content: '{"temp": 18}', tool_call_id: id },
  ]
  const bad = lintFor('mistral03', { messages: calls('call_1'), tools: weather })
  assert.equal(bad[0].id, 'raise')
  assert.match(bad[0].title, /length 9/)
  assert.ok(render(input('mistral03', { messages: calls('abc123xyz'), tools: weather })).ok)
})

test('Mistral v0.3 prefix mismatch is attributed to the moved system prompt', () => {
  const msgs: ChatMessage[] = [...chat, { role: 'assistant', content: 'Hello.' }, { role: 'user', content: 'More?' }]
  const p = lintFor('mistral03', { messages: msgs }).find((i) => i.id === 'prefix-mismatch')
  assert.ok(p)
  assert.match(p.detail, /system prompt lands in a different turn/)
})

// Expected values produced by Jinja2 3.1.6 configured as in transformers' chat_template_utils.py.
test('`| string` and raise_exception are only hooked where they are code', () => {
  const r = (template: string, kwargs: Record<string, unknown> = {}) => {
    const out = render({ template, messages: [], addGenerationPrompt: false, kwargs })
    return out.ok ? out.text : `${out.error.kind}: ${out.error.message}`
  }
  assert.equal(r('Return: number | string.'), 'Return: number | string.')
  assert.equal(r("{{ 'union: number | string' }}"), 'union: number | string')
  assert.equal(r('{{ "call raise_exception(\'x\')" }} raise_exception(now)'), "call raise_exception('x') raise_exception(now)")
  assert.equal(r('{# a | string #}ok'), 'ok')
  assert.equal(r('{{ nothing | string }}|{{ x.nothing | string }}|', { x: {} }), '||')
  assert.equal(r('{{ x | string }}', { x: null }), 'None')
  assert.equal(r('{{ (x) | string }}', { x: { a: 1 } }), "{'a': 1}")
  assert.equal(r("{{ {'a': 1} | string }} {{ [1, 2] | string }} {{ 5 | string }} {{ 1.0 | string }}"), "{'a': 1} [1, 2] 5 1.0")
  assert.equal(r('{{ x | list | string }}', { x: ['a', true] }), "['a', True]")
  assert.equal(r('{{ x | string | upper }}{{ x | string() }}', { x: { a: 1 } }), "{'A': 1}{'a': 1}")
  assert.equal(r("{% set y = x | string %}{% if x | string == '[1]' %}{{ 'p' + y }}{% endif %}", { x: [1] }), 'p[1]')
  assert.equal(r('{% macro f(v) %}{{ v | string }}{% endmacro %}{{ f(x) }}', { x: [1] }), '[1]')
  assert.equal(r("{{ {'k': x | string}['k'] }}", { x: [1] }), '[1]')
  assert.equal(r("{% if x is string %}s{% endif %}", { x: 'a' }), 's')
  assert.equal(r("{{ raise_exception('no ' + x) }}", { x: 'way' }), 'raise: no way')
})

test('a tool or call counts as rendered only when the prompt follows its name', () => {
  const CHAT = '{% for m in messages %}<|{{ m.role }}|>{{ m.content }}<|end|>{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}'
  const call = (name: string) => ({ id: 'abc123xyz', type: 'function', function: { name, arguments: { city: 'Paris' } } })
  const trip = (...calls: unknown[]): ChatMessage[] => [
    { role: 'user', content: 'use get_weather' },
    { role: 'assistant', content: '', tool_calls: calls },
    { role: 'tool', content: '21', tool_call_id: 'abc123xyz' },
  ]
  const on = (template: string, messages: ChatMessage[], tools: unknown[] = weather) => ids(lint({ template, input: { messages, tools, addGenerationPrompt: true }, tokenizerAddsBos: false }))

  // Schema rendered, call dropped.
  const schemaOnly = on('{{ tools | tojson }}' + CHAT, trip(call('get_weather')))
  assert.ok(schemaOnly.includes('tools') && schemaOnly.includes('calls-1'))
  // Tools change the prompt, but the schema is only "there" because the user wrote the name.
  const nameOnly = on('{% if tools %}You have tools.{% endif %}' + CHAT, [{ role: 'user', content: 'use get_weather' }])
  assert.ok(nameOnly.includes('tools-partial') && !nameOnly.includes('tools'))
  // Only the first of two calls with the same name is rendered.
  assert.ok(on('{% for m in messages %}{% if m.tool_calls %}{{ m.tool_calls[0].function.name }}{% endif %}{{ m.content }}{% endfor %}', trip(call('f'), call('f')), []).includes('calls-1'))
  // Only the first of two tools with the same name is rendered; a nameless tool is never called rendered.
  assert.ok(on('{% if tools %}{{ tools[0] | tojson }}{% endif %}' + CHAT, trip(call('x')), [...weather, ...weather]).includes('tools-partial'))
  assert.ok(on('{{ tools | tojson }}' + CHAT, trip(call('x')), [{}]).includes('tools-partial'))
  // A template that refuses calls to undeclared tools has read the name.
  const strict = "{{ tools | tojson }}{% for m in messages %}{% for c in m.tool_calls or [] %}{% if c.function.name != 'get_weather' %}{{ raise_exception('unknown tool') }}{% endif %}{{ c.function.name }}{% endfor %}{{ m.content }}{% endfor %}"
  assert.ok(!on(strict, trip(call('get_weather'))).includes('calls-1'))
  for (const name of ['qwen25', 'qwen3', 'llama31', 'mistral03']) {
    const i = ids(lintFor(name, { messages: trip(call('get_weather')), tools: weather }))
    assert.ok(i.includes('tools') && !i.includes('calls-1'), `${name}: ${i}`)
  }
  // SmolLM3 renders schemas but its template never reads tool_calls.
  const smol = ids(lintFor('smollm3', { messages: trip(call('get_weather')), tools: weather }))
  assert.ok(smol.includes('tools') && smol.includes('calls-1'))
})

// Expected values produced by Jinja2 3.1.6 with the tojson filter of transformers' chat_template_utils.py (Python 3.10).
test('JSON numbers render as Python has them', () => {
  const r = (template: string, json: string) => {
    const out = render({ template, messages: [], addGenerationPrompt: false, kwargs: parsePython(json) as Record<string, unknown> })
    return out.ok ? out.text : `${out.error.kind}: ${out.error.message}`
  }
  const N = '{"x": [1.0, 12345678901234567890, 1e-7, 1e16, 1e22, 0.1, -0.0, -0, 1E3, 100000.0, 0.0001, 0.00001, 9007199254740993, 1e999, -5, 1.50]}'
  assert.equal(r('{{ x | tojson }}', N), '[1.0, 12345678901234567890, 1e-07, 1e+16, 1e+22, 0.1, -0.0, 0, 1000.0, 100000.0, 0.0001, 1e-05, 9007199254740993, Infinity, -5, 1.5]')
  assert.equal(r('{{ x | string }}', N), '[1.0, 12345678901234567890, 1e-07, 1e+16, 1e+22, 0.1, -0.0, 0, 1000.0, 100000.0, 0.0001, 1e-05, 9007199254740993, inf, -5, 1.5]')
  assert.equal(r('{{ x }}', N), r('{{ x | string }}', N))
  assert.equal(r('{% for n in x %}{{ n }} {% endfor %}', N), '1.0 12345678901234567890 1e-07 1e+16 1e+22 0.1 -0.0 0 1000.0 100000.0 0.0001 1e-05 9007199254740993 inf -5 1.5 ')
  assert.equal(r("{{ 't=' ~ x[0] ~ x[1] ~ x[2] }}", N), 't=1.0123456789012345678901e-07')
  assert.equal(r("{{ x | join('/') }}", '{"x": [1.0, 2, "a", true, null, 1e-7]}'), '1.0/2/a/True/None/1e-07')
  assert.equal(r('{{ x }}', '{"x": {"a": [1.0, "b", null], "c": true}}'), "{'a': [1.0, 'b', None], 'c': True}")
  assert.equal(r('{{ a + 1 }} {{ 1 if a == 1 else 0 }} {{ 1 if a is integer else 0 }} {{ 1 if z else 0 }}', '{"a": 1.0, "z": 0.0}'), '2.0 1 0 0')
  assert.equal(r('{{ x | tojson(indent=2) }}', '{"x": {"b": [1.0, {}], "a": "é\\u007f"}}'), '{\n  "b": [\n    1.0,\n    {}\n  ],\n  "a": "é\u007f"\n}')
  assert.equal(r("{{ x | tojson(sort_keys=true, ensure_ascii=true, separators=(',', ':')) }}", '{"x": {"b": 1.0, "a": "é"}}'), '{"a":"\\u00e9","b":1.0}')
  assert.equal(r('{{ 1.0 | tojson }} {{ 2 | tojson }}', '{}'), '1.0 2')
  // Computing with an integer JS can't hold is an error, not another number.
  assert.match(r('{{ x + 1 }}', '{"x": 12345678901234567890}'), /^runtime: .*9,007,199,254,740,991/)
})

test('numbers in tools and tool_calls, and text that looks like numbers', () => {
  const tools = parsePython('[{"name": "f", "parameters": {"minimum": 1.0, "maximum": 12345678901234567890, "note": "1.0 \\"2.0\\" 3e5"}}]') as unknown[]
  const out = render({ template: '{{ tools | tojson }}{{ messages[0].tool_calls[0].arguments | tojson }}', messages: [{ role: 'assistant', content: '', tool_calls: parsePython('[{"arguments": {"t": 20.0}}]') as unknown[] }], tools, addGenerationPrompt: false })
  assert.deepEqual(out, { ok: true, text: '[{"name": "f", "parameters": {"minimum": 1.0, "maximum": 12345678901234567890, "note": "1.0 \\"2.0\\" 3e5"}}]{"t": 20.0}' })
  for (const bad of ['[1.]', '[01]', '{"a": 1.0', '1.0.0', '[NaN]', '']) assert.throws(() => parsePython(bad), SyntaxError, bad)
  for (const ok of ['{"a": [1, 2.5, "x", null, true, {"b": -3}]}', '[]', '"1.0"', '1.5']) assert.deepEqual(parsePython(ok), JSON.parse(ok))
})

test('a prompt that starts with BOS twice is an error whatever the tokenizer adds', () => {
  const C = '{% for m in messages %}[{{ m.role }}]{{ m.content }}{% endfor %}'
  const on = (template: string, adds: boolean | null, bos: string | undefined, messages: ChatMessage[] = [{ role: 'user', content: 'a' }, { role: 'assistant', content: 'b' }, { role: 'user', content: 'c' }]) =>
    lint({ template, input: { messages, addGenerationPrompt: true, specials: { bos_token: bos } }, tokenizerAddsBos: adds }).filter((i) => /bos/.test(i.id))
  for (const adds of [true, false, null]) {
    for (const t of ['{{ bos_token }}{{ bos_token }}' + C, '{{ bos_token + bos_token }}' + C, '<s>{{ bos_token }}' + C]) {
      const found = on(t, adds, '<s>')
      assert.deepEqual(found.map((i) => [i.id, i.severity]), [['double-bos', 'error']], `${t} ${adds}`)
    }
  }
  // BOS from the first message counts too: it is tokenized the same.
  assert.equal(on('{{ bos_token }}{{ messages[0].content }}', false, '<s>', [{ role: 'user', content: '<s>hi' }])[0].severity, 'error')
  // Llama 2 repeats BOS before every later user turn.
  const llama2 = "{% for m in messages %}{% if m.role == 'user' %}{{ bos_token + '[INST] ' + m.content + ' [/INST]' }}{% else %}{{ ' ' + m.content + ' </s>' }}{% endif %}{% endfor %}"
  assert.deepEqual(on(llama2, false, '<s>').map((i) => i.id), ['bos'])
  assert.deepEqual(on(llama2, true, '<s>').map((i) => i.severity), ['error'])
  assert.deepEqual(on('{{ bos_token }}' + C, null, '<s>').map((i) => i.id), ['bos-unknown'])
  assert.deepEqual(on(C, null, undefined), [])
  assert.deepEqual(on(C, null, ''), [])
})

test('a raise about the system message gets the advice its text calls for', () => {
  const late: ChatMessage[] = [{ role: 'user', content: 'a' }, { role: 'system', content: 'b' }]
  const hint = (message: string, messages = late) =>
    lint({ template: `{{ raise_exception(${JSON.stringify(message)}) }}`, input: { messages, addGenerationPrompt: true }, tokenizerAddsBos: null })[0].detail
  for (const m of ['System role not supported', 'System role not supported!', 'System messages are not supported.', 'system role is not supported by this model', 'This model does not support system messages', "This template doesn't accept a system prompt.", 'No system role', 'System prompt not allowed'])
    assert.match(hint(m), /no system role\. Fold/, m)
  for (const m of ['System message must be at the beginning.', 'Only user and assistant roles are supported, with the exception of an initial optional system message!', 'System role must be first', 'System messages are only supported at the start of the conversation', 'System message not supported after the first turn', 'System message is not allowed in the middle of a conversation; put it first'])
    assert.match(hint(m), /only at the start.*Move message #2 there\./, m)
  assert.match(hint('System message must be at the beginning.', [{ role: 'system', content: 'x' }, ...late]), /Move message #3 there, merged into the first one/)
  assert.match(hint('Only user and assistant roles are supported, with the exception of an initial optional system message!', [{ role: 'user', content: 'a' }, { role: 'tool', content: 'b' }]), /leaves out the tool messages/)
  // Neither: no advice that could be wrong.
  for (const m of ['System message cannot contain images.', 'System message must be a string', 'Unexpected item in system content'])
    assert.match(hint(m), /refuses this conversation shape\.$/, m)
  assert.match(hint('After the optional system message, conversation roles must alternate user/assistant'), /Merge consecutive/)
})

test('"never read" follows what the template reads, not the words in it', () => {
  const C = '{% for m in messages %}{{ m.role }}{{ m.content }}{% endfor %}'
  const unread = (template: string, kwargs: Record<string, unknown> = { x: 1 }) =>
    lint({ template, input: { messages: [{ role: 'user', content: 'hi' }], addGenerationPrompt: false, kwargs }, tokenizerAddsBos: null }).filter((i) => i.id.startsWith('kwarg-')).map((i) => i.id.slice(6))
  for (const t of ['{# x #}', 'x', "{{ 'x' }}", '{{ messages[0].x }}', "{{ messages[0]['x'] }}", '{% set x = 1 %}{{ x }}', '{% for x in messages %}{{ x.role }}{% endfor %}', '{% macro f(x) %}{{ x }}{% endmacro %}{{ f(1) }}', '{% set ns = namespace(x=1) %}{{ ns.x }}', '{% set ns = namespace(y=1) %}{% set ns.x = 2 %}', '{% if messages %}{% set x = 1 %}{% else %}{% set x = 2 %}{% endif %}{{ x }}', '{% for m in messages if m.x %}a{% endfor %}', '{% set x %}q{% endset %}{{ x }}', "{{ 'a' is string }}", "{% set x, y = ['a', 'b'] %}{{ x }}"])
    assert.deepEqual(unread(t + C), ['x'], t)
  for (const t of ['{{ x }}', '{% if x is defined %}a{% endif %}', '{% if x is not defined %}{% set x = true %}{% endif %}{{ x }}', '{% set x = x | default(1) %}', '{{ y | default(x) }}', '{% if messages %}{% set x = 1 %}{% endif %}{{ x }}', '{{ messages[x - 1] }}', '{{ messages[x:] }}', "{{ {'k': x} }}", '{{ "a" if x else "b" }}', '{{ "a" if x }}', '{% for m in messages if x %}a{% endfor %}', '{% for m in messages %}{% set x = 1 %}{% endfor %}{{ x }}', '{% macro f(a=x) %}{{ a }}{% endmacro %}', '{% macro f() %}{{ x }}{% endmacro %}{% set x = 2 %}', '{% if not x %}a{% endif %}', '{{ "a" ~ x }}', '{{ x | tojson }}', '{{ x.y }}', '{% set ns = namespace(y=x) %}', '{% filter upper %}{{ x }}{% endfilter %}', '{% for x in x %}a{% endfor %}', '{% if 1 in [x] %}a{% endif %}', '{{ raise_exception(x) if false }}'])
    assert.deepEqual(unread(t + C), [], t)
  assert.deepEqual(unread(C, { role: 'x', content: 1, messages_: 2 }), ['role', 'content', 'messages_'])
  for (const [name, k] of [['qwen3', 'enable_thinking'], ['llama31', 'date_string'], ['llama31', 'tools_in_user_message'], ['smollm3', 'enable_thinking']])
    assert.ok(!ids(lintFor(name, { kwargs: { [k]: 'x' } })).includes(`kwarg-${k}`), `${name} ${k}`)
})
