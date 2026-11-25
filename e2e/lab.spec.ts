import { test, expect, openLab, paste, conversation, prompt, issue, setSwitch, loadModel, stored, tab } from './fixtures.ts'

const CHATML = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
const sample = (page: import('@playwright/test').Page, label: string) => page.getByLabel('Load a sample conversation').selectOption({ label })

test('conversation editor: add, change role, move, delete, and reload', async ({ page }) => {
  await openLab(page)
  await paste(page, '{% for m in messages %}{{ m.role }}={{ m.content }};{% endfor %}')
  await conversation(page, [{ role: 'user', content: 'one' }, { role: 'assistant', content: 'two' }])
  await setSwitch(page, 'add_generation_prompt', false)
  await expect.poll(() => prompt(page)).toBe('user=one;assistant=two;')
  await page.getByRole('button', { name: 'Move down' }).first().click()
  await expect.poll(() => prompt(page)).toBe('assistant=two;user=one;')
  await page.getByLabel('Role of message 1').selectOption('system')
  await expect.poll(() => prompt(page)).toBe('system=two;user=one;')
  await page.reload()
  await expect.poll(() => prompt(page)).toBe('system=two;user=one;')
  await page.getByRole('button', { name: 'Delete message 1' }).click()
  await expect.poll(() => prompt(page)).toBe('user=one;')
})

test('multi-turn sample on ChatML is clean', async ({ page }) => {
  await openLab(page)
  await sample(page, 'Multi-turn chat')
  await paste(page, CHATML)
  await expect(page.locator('.card.lint .counts')).toHaveText('clean')
  await expect(issue(page, 'Generation prompt appended')).toContainText('"<|im_start|>assistant\\n"')
})

test('tool sample: Phi-3.5 drops tools, call and result; Qwen2.5 renders them', async ({ page }) => {
  await openLab(page)
  await sample(page, 'Tool-call round trip')
  await expect(issue(page, 'Tools rendered (1)')).toBeVisible()
  await expect.poll(() => prompt(page)).toContain('<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>')
  await loadModel(page, 'microsoft/Phi-3.5-mini-instruct')
  await expect(issue(page, 'Tools are ignored by this template')).toBeVisible()
  await expect(issue(page, 'Message #4 (tool) is dropped')).toBeVisible()
  await expect(issue(page, 'tool_calls in message #3 are dropped')).toBeVisible()
  await expect.poll(() => prompt(page)).not.toContain('Paris"}')
})

test('broken alternation sample is refused by Mistral and nothing can be copied', async ({ page }) => {
  await openLab(page)
  await sample(page, 'Broken alternation')
  await expect(page.locator('.source .meta')).toContainText('mistralai/Mistral-7B-Instruct-v0.3')
  await expect(issue(page, /^Template raised: .*alternate/)).toContainText('Merge consecutive same-role messages')
  await tab(page, 'Rendered')
  await expect(page.locator('.render-error')).toContainText('raise_exception')
  await expect(page.getByRole('button', { name: 'Copy', exact: true })).toHaveCount(0)
})

test('reasoning sample: enable_thinking reaches Qwen3 and is reported unused by Qwen2.5', async ({ page }) => {
  await openLab(page)
  await sample(page, 'Reasoning off (enable_thinking)')
  await expect(page.locator('.source .meta')).toContainText('Qwen/Qwen3-8B')
  await expect.poll(() => prompt(page)).toBe('<|im_start|>user\nIs 9.11 bigger than 9.9?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
  await expect(issue(page, /enable_thinking/)).toHaveCount(0)
  await loadModel(page, 'Qwen/Qwen2.5-7B-Instruct')
  await expect(issue(page, 'Kwarg `enable_thinking` is never read')).toBeVisible()
})

test('the kwarg shortcut fills in what the template reads', async ({ page }) => {
  await openLab(page)
  await loadModel(page, 'Qwen/Qwen3-8B')
  await page.getByRole('button', { name: 'enable_thinking' }).click()
  expect(JSON.parse(await page.getByLabel('Extra kwargs').inputValue())).toEqual({ enable_thinking: false })
})

test('linter: dropped system message, ignored generation prompt, trailing space, prefix mismatch', async ({ page }) => {
  await openLab(page)
  await conversation(page, [{ role: 'system', content: 'Rules' }, { role: 'user', content: 'Hi' }])
  await paste(page, "{% for m in messages %}{% if m.role != 'system' %}{{ m.role }}: {{ m.content }}\n{% endif %}{% endfor %}")
  await expect(issue(page, 'System message silently dropped')).toBeVisible()
  await expect(issue(page, 'add_generation_prompt is ignored')).toBeVisible()
  await page.getByLabel('Chat template source').fill('{% for m in messages %}[{{ m.role }}] {{ m.content }}\n{% endfor %}{% if add_generation_prompt %}[assistant]<think> {% endif %}')
  await expect(issue(page, 'Generation prompt ends with a space')).toBeVisible()
  await expect(issue(page, 'Inference prompt diverges from training format')).toBeVisible()
  await expect(issue(page, 'System message silently dropped')).toHaveCount(0)
})

test('template that does not parse is an error with no prompt', async ({ page }) => {
  await openLab(page)
  await paste(page, '{% for m in messages %}{{ m.content }}')
  await expect(issue(page, 'Template does not parse')).toBeVisible()
  await tab(page, 'Rendered')
  await expect(page.locator('.render-error')).toContainText('syntax error')
  await expect(page.getByRole('button', { name: 'Copy', exact: true })).toHaveCount(0)
})

test('an empty pasted template is reported as missing', async ({ page }) => {
  await openLab(page)
  await paste(page, '   ')
  await expect(issue(page, 'No chat template')).toHaveClass(/sev-error/)
})

test('markup in messages is shown as text', async ({ page }) => {
  await openLab(page)
  await paste(page, '{% for m in messages %}{{ m.content }}{% endfor %}')
  await conversation(page, [{ role: 'user', content: '<img src=x onerror="window.__pwned=1"><b>bold</b>' }])
  await tab(page, 'Rendered')
  await expect(page.locator('.rendered')).toContainText('<b>bold</b>')
  await expect(page.locator('.rendered img, .rendered b')).toHaveCount(0)
  expect(await page.evaluate(() => (window as unknown as { __pwned?: number }).__pwned)).toBeUndefined()
})

test('whitespace marks can be turned on and off', async ({ page }) => {
  await openLab(page)
  await paste(page, 'a  b\n\tc')
  await tab(page, 'Rendered')
  await expect(page.locator('.rendered .ws')).toHaveText(['··​', '↵', '→\t'])
  await page.getByLabel('whitespace').uncheck()
  await expect(page.locator('.rendered')).toHaveText('a  b\n\tc')
  await expect(page.locator('.rendered .ws')).toHaveCount(0)
})

test('invalid JSON is named next to the options', async ({ page }) => {
  await openLab(page)
  await setSwitch(page, 'tools', true)
  await page.getByLabel('Tools JSON').fill('{}')
  await page.getByLabel('Extra kwargs').fill('[1]')
  await expect(page.locator('.input-errors').getByText('Tools must be a JSON array')).toBeVisible()
  await expect(page.locator('.input-errors').getByText('Extra kwargs must be a JSON object')).toBeVisible()
})

test('tools, kwargs and tool_calls survive a reload', async ({ page }) => {
  await openLab(page)
  await paste(page, '{{ tools | length }}|{{ mode }}|{{ messages[0].tool_calls[0].function.name }}')
  await conversation(page, [{ role: 'assistant', content: '', toolCalls: '[{"function":{"name":"f1"}}]' }])
  await setSwitch(page, 'tools', true)
  await page.getByLabel('Tools JSON').fill('[{"name":"a"},{"name":"b"}]')
  await page.getByLabel('Extra kwargs').fill('{"mode":"fast"}')
  await expect.poll(() => prompt(page)).toBe('2|fast|f1')
  await page.reload()
  await expect.poll(() => prompt(page)).toBe('2|fast|f1')
  expect((await stored(page))?.model).toBe('')
})

test('each kind of unreadable input withholds the prompt, tokens and comparison until it is fixed', async ({ page }) => {
  await openLab(page)
  await loadModel(page, 'test/tiny')
  await conversation(page, [{ role: 'user', content: 'hi' }, { role: 'assistant', content: '', toolCalls: '[]' }])
  await setSwitch(page, 'tools', true)
  await page.getByLabel('Tools JSON').fill('')
  await tab(page, 'Tokens')
  await page.getByRole('button', { name: 'Load tokenizer' }).click()
  await expect(page.locator('.stat-row')).toBeVisible()
  const bad = issue(page, 'Input can’t be read')
  for (const [label, text, message, good] of [
    ['Tools JSON', '[{"type":', /Tools: /, ''],
    ['Tools JSON', '{}', /Tools must be a JSON array/, '[]'],
    ['Tools JSON', '[{"name":"a"}, "x", null, []]', /Tools: item #2 must be a JSON object/, '[{"name":"a"}]'],
    ['Extra kwargs', '{"a": fal', /Extra kwargs: /, ' '],
    ['Extra kwargs', 'null', /Extra kwargs must be a JSON object/, '{}'],
    ['Message 2 tool_calls JSON', '[{"function"', /Message #2 tool_calls: /, '[]'],
    ['Message 2 tool_calls JSON', '{}', /Message #2 tool_calls must be a JSON array/, ''],
  ] as const) {
    await page.getByLabel(label).fill(text)
    await expect(bad.first()).toContainText(message)
    await expect(page.locator('.card.lint .sev-ok')).toHaveCount(0)
    await expect(page.locator('.stat-row')).toHaveCount(0)
    await page.getByLabel(label).fill(good)
    await expect(bad).toHaveCount(0)
    await expect(page.locator('.stat-row')).toBeVisible()
  }
  await page.getByLabel('Extra kwargs').fill('{')
  await tab(page, 'Rendered')
  await expect(page.locator('.render-error')).toContainText('unreadable input')
  await tab(page, 'Compare models')
  await page.getByLabel('Compare with model').fill('test/tiny-long')
  await page.getByRole('button', { name: 'Compare', exact: true }).click()
  await expect(page.locator('.cmp-delta')).toContainText('No delta: the input can’t be read')
  await expect(page.locator('.cmp-grid .rendered')).toHaveCount(0)
})

const UNSAVED = 'This browser can’t save your setup'

test('a setup over the storage quota renders, says it is not saved, and recovers', async ({ page }) => {
  await openLab(page)
  await paste(page, '{{ messages[0].content | length }}')
  // Leave less than 100,000 characters of the 5 MB quota, so the next message does not fit.
  await page.evaluate(() => {
    for (let step = 1 << 22; step >= 1 << 16; step /= 2) {
      try {
        localStorage.setItem('filler', (localStorage.getItem('filler') ?? '') + 'x'.repeat(step))
      } catch { /* full: try a smaller step */ }
    }
  })
  await page.getByLabel('Message 1 content').fill('x'.repeat(200_000))
  await expect(page.getByText(UNSAVED)).toBeVisible()
  await expect.poll(() => prompt(page)).toBe('200000')
  await page.getByLabel('Message 1 content').fill('short')
  await expect(page.getByText(UNSAVED)).toHaveCount(0)
  expect((await stored(page))?.msgs[0].content).toBe('short')
  await page.reload()
  await expect(page.getByLabel('Message 1 content')).toHaveValue('short')
})

test('the app works when storage is blocked altogether', async ({ page }) => {
  await page.addInitScript(() => Object.defineProperty(window, 'localStorage', { get() { throw new DOMException('denied', 'SecurityError') } }))
  await page.goto('/')
  await page.getByRole('button', { name: 'Switch to dark theme' }).click()
  await expect(page.locator('html')).toHaveAttribute('data-theme-mode', 'dark')
  await openLab(page)
  await expect(page.getByText(UNSAVED)).toBeVisible()
  await paste(page, 'ok')
  await expect.poll(() => prompt(page)).toBe('ok')
})
