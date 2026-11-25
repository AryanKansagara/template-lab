import { test, expect, openLab, paste, conversation, prompt, issue, setSwitch, loadModel, stored, tab, shareHash } from './fixtures.ts'

// Each test states the right behaviour of a filed bug. Those marked test.fail are not fixed yet.

const CHAT = '{% for m in messages %}<|{{ m.role }}|>{{ m.content }}<|end|>{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}'
const counts = '.card.lint .counts'

test('#2 "| string" in plain text and literals is left alone', async ({ page }) => {
  await openLab(page)
  await paste(page, "Return: number | string. {{ 'union: number | string' }}")
  await expect.poll(() => prompt(page)).toBe('Return: number | string. union: number | string')
})

test('#3 a dropped tool call is reported although the schema names the tool', async ({ page }) => {
  await openLab(page)
  await page.getByLabel('Load a sample conversation').selectOption({ label: 'Tool-call round trip' })
  await paste(page, '{% if tools %}{{ tools | tojson }}{% endif %}' + CHAT)
  await expect(issue(page, 'Tools rendered (1)')).toBeVisible()
  await expect(issue(page, 'tool_calls in message #3 are dropped')).toBeVisible()
})

test('#4 unreadable tools JSON gives no prompt and no clean verdict', async ({ page }) => {
  await openLab(page)
  await setSwitch(page, 'tools', true)
  await page.getByLabel('Tools JSON').fill('[{"type":')
  await expect(page.locator('.input-errors')).toBeVisible()
  await expect(page.locator(counts)).not.toContainText('clean')
  await expect(page.getByRole('button', { name: 'Copy', exact: true })).toHaveCount(0)
})

test('#5 the lab keeps working when the state cannot be stored', async ({ page }) => {
  await page.addInitScript(() => { Storage.prototype.setItem = () => { throw new DOMException('full', 'QuotaExceededError') } })
  await openLab(page)
  await expect(page.getByRole('heading', { name: 'Conversation' })).toBeVisible()
})

test('#6 a malformed preset link opens a working lab', async ({ page }) => {
  await openLab(page, '#try=chat|%')
  await expect(page.getByRole('heading', { name: 'Conversation' })).toBeVisible()
})

test('#6 a share link that is not a setup keeps the saved work', async ({ page }) => {
  await openLab(page)
  await paste(page, 'MINE')
  await expect.poll(async () => (await stored(page))?.override).toBe('MINE')
  await page.goto('about:blank')
  await page.goto('/app' + (await shareHash(5)))
  await expect(page.getByRole('status')).toHaveText('That share link is broken or truncated.')
  expect((await stored(page))?.override).toBe('MINE')
})

test('#7 the revision in a Hub URL is the one that is loaded', async ({ page, hub }) => {
  hub.files['test/jinja-wins@v1.0/chat_template.jinja'] = { body: 'V1' }
  await openLab(page)
  await loadModel(page, 'https://huggingface.co/test/jinja-wins/tree/v1.0')
  await expect(page.locator('.source .meta')).toContainText('test/jinja-wins')
  await expect.poll(() => prompt(page)).toBe('V1')
})

test('#8 tool_use is used when tools are on', async ({ page }) => {
  await openLab(page)
  await loadModel(page, 'test/named')
  await setSwitch(page, 'tools', true)
  await expect.poll(() => prompt(page), { timeout: 3000 }).toBe('TOOLS:1')
})

test('#9 a truncated chat_template.json is reported as unreadable', async ({ page }) => {
  await openLab(page)
  await loadModel(page, 'test/truncated-processor')
  await expect(page.locator('.hub-error')).toContainText('chat_template.json is not valid JSON', { timeout: 3000 })
})

test('#10 a template that writes BOS twice is an error', async ({ page }) => {
  await openLab(page)
  await paste(page, '{{ bos_token }}{{ bos_token }}' + CHAT, '<s>')
  expect((await prompt(page)).startsWith('<s><s>')).toBe(true)
  await expect(issue(page, 'Duplicate BOS token')).toBeVisible({ timeout: 3000 })
})

test('#11 a kwarg that only appears in a comment is reported as never read', async ({ page }) => {
  await openLab(page)
  await paste(page, '{# enable_thinking #}' + CHAT)
  await page.getByLabel('Extra kwargs').fill('{"enable_thinking": false}')
  await expect(issue(page, 'Kwarg `enable_thinking` is never read')).toBeVisible({ timeout: 3000 })
})

test.fail('#12 a pasted template survives a visit to the Hub tab', async ({ page }) => {
  await openLab(page)
  await paste(page, 'MINE')
  await page.getByRole('tab', { name: 'From the Hub' }).click()
  await page.getByRole('tab', { name: 'Paste a template' }).click()
  await expect(page.getByLabel('Chat template source')).toHaveValue('MINE', { timeout: 3000 })
})

test('#13 a system message in the wrong place is not called a missing system role', async ({ page }) => {
  await openLab(page)
  await conversation(page, [{ role: 'user', content: 'a' }, { role: 'system', content: 'b' }])
  await paste(page, "{% for m in messages %}{% if m.role == 'system' and not loop.first %}{{ raise_exception('System message must be at the beginning.') }}{% endif %}{% endfor %}" + CHAT)
  await expect(issue(page, 'Template raised')).toBeVisible()
  await expect(issue(page, 'Template raised')).not.toContainText('no system role', { timeout: 3000 })
})

test('#14 JSON numbers keep their Python form', async ({ page }) => {
  await openLab(page)
  await paste(page, '{{ x | tojson }}')
  await page.getByLabel('Extra kwargs').fill('{"x": [1.0, 12345678901234567890]}')
  await expect.poll(() => prompt(page), { timeout: 3000 }).toBe('[1.0, 12345678901234567890]')
})

test.fail('#15 a refused copy is not reported as copied', async ({ page }) => {
  await openLab(page)
  await paste(page, 'x')
  await tab(page, 'Rendered')
  await page.evaluate(() => Object.defineProperty(navigator, 'clipboard', { value: { writeText: () => Promise.reject(new Error('denied')) }, configurable: true }))
  await page.getByRole('button', { name: 'Copy', exact: true }).click()
  await expect(page.getByRole('status')).not.toHaveText('Rendered prompt copied', { timeout: 1000 })
})

// #16 (comparing 20,000 lines freezes the tab) has no test here: pinning it would hang a worker for minutes.

test.fail('#17 the kwarg shortcut does not erase invalid kwargs', async ({ page }) => {
  await openLab(page)
  await paste(page, '{% if foo is defined %}{% endif %}x')
  await page.getByLabel('Extra kwargs').fill('{"a": 1,}')
  await page.getByRole('button', { name: 'foo', exact: true }).click()
  await expect(page.getByLabel('Extra kwargs')).toHaveValue(/"a": 1/, { timeout: 3000 })
})
