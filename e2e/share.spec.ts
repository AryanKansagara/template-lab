import { test, expect, openLab, paste, conversation, prompt, copied, stored, shareHash } from './fixtures.ts'

async function ownWork(page: import('@playwright/test').Page) {
  await openLab(page)
  await paste(page, 'MINE:{{ messages[0].content }}')
  await conversation(page, [{ role: 'user', content: 'keep me' }])
  await expect.poll(async () => (await stored(page))?.msgs.length).toBe(1)
  return page.evaluate(() => localStorage.getItem('tl:state'))
}

test('a share link opens the same setup in a fresh browser', async ({ page, browser, baseURL }) => {
  await openLab(page)
  await paste(page, '{{ bos_token }}{{ messages[0].content }}{{ flag }}', '<s>')
  await conversation(page, [{ role: 'user', content: 'héllo 🤗 <|im_end|> <script>' }])
  await page.getByLabel('Extra kwargs').fill('{"flag": "é"}')
  await expect.poll(() => prompt(page)).toBe('<s>héllo 🤗 <|im_end|> <script>é')
  await page.getByRole('button', { name: 'Share' }).click()
  await expect(page.getByRole('status')).toHaveText('Share link copied to clipboard')
  const link = await copied(page)
  expect(link).toMatch(/\/app#s=[\w-]+$/)
  const saved = await stored(page)

  const other = await browser.newContext({ baseURL })
  await other.route('**/*', (r) => (new URL(r.request().url()).origin === new URL(baseURL!).origin ? r.continue() : r.abort()))
  const p2 = await other.newPage()
  const errors: string[] = []
  p2.on('pageerror', (e) => errors.push(String(e)))
  await p2.goto(link)
  await expect.poll(() => stored(p2)).toEqual(saved)
  // The setup leaves the address bar once applied.
  await expect(p2).toHaveURL(/\/app$/)
  expect(errors).toEqual([])
  await other.close()
})

for (const [name, hash] of [
  ['garbage', '#s=garbage!!'],
  ['not deflate', '#s=AAAA'],
  ['JSON null', null],
]) {
  test(`broken share link (${name}) says so and keeps the saved work`, async ({ page }) => {
    const before = await ownWork(page)
    await page.goto('about:blank')
    await page.goto('/app' + (hash ?? (await shareHash(null))))
    await expect(page.getByRole('status')).toHaveText('That share link is broken or truncated.')
    expect(await page.evaluate(() => localStorage.getItem('tl:state'))).toBe(before)
  })
}

test('a truncated share link says so and keeps the saved work', async ({ page }) => {
  const before = await ownWork(page)
  const full = await shareHash({ model: '', override: 'OTHER', msgs: [{ role: 'user', content: 'some text that is long enough to be cut' }] })
  await page.goto('about:blank')
  await page.goto('/app' + full.slice(0, -20))
  await expect(page.getByRole('status')).toHaveText('That share link is broken or truncated.')
  expect(await page.evaluate(() => localStorage.getItem('tl:state'))).toBe(before)
})

test('a share link with wrong-typed fields falls back field by field', async ({ page }) => {
  await page.goto('/app' + (await shareHash({ model: 7, override: 'OK', kwargs: {}, agp: 'yes', msgs: [{ role: 'user', content: 'x' }], evil: 1 })))
  await expect.poll(async () => (await stored(page))?.override).toBe('OK')
  const s = (await stored(page))!
  expect(s.model).toBe('Qwen/Qwen2.5-7B-Instruct')
  expect(s.kwargs).toBe('{}')
  expect(s.agp).toBe(true)
  expect(Object.keys(s)).not.toContain('evil')
})

test('corrupt saved state opens the defaults', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('tl:state', '{"model": "Qwen'))
  await openLab(page)
  await expect(page.getByLabel('Model id')).toHaveValue('Qwen/Qwen2.5-7B-Instruct')
})

test('landing presets open the lab with the sample and the model', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('link', { name: 'Try a tool-call render' }).click()
  await expect(page.locator('.source .meta')).toContainText('Qwen/Qwen2.5-7B-Instruct')
  await expect(page.getByText('4 messages')).toBeVisible()
  expect((await stored(page))?.toolsOn).toBe(true)
  await page.goto('/')
  await page.getByRole('link', { name: /gemma-2-2b-it/ }).click()
  await expect(page.locator('.card.lint')).toContainText('Template raised: System role not supported')
})

const BROKEN = 'That share link is broken or truncated.'
const UNREADABLE = 'That share link has messages this lab can’t read, so it was not opened.'
const ok = { model: '', override: 'OTHER' }
for (const [name, payload, message] of [
  ['a number', 5, BROKEN],
  ['a string', 'x', BROKEN],
  ['a list', [1], BROKEN],
  ['an empty object', {}, BROKEN],
  ['only unknown fields', { evil: 1, __proto__x: {} }, BROKEN],
  ['messages that are not a list', { ...ok, msgs: 'hi' }, UNREADABLE],
  ['an unknown role', { ...ok, msgs: [{ role: 'developer', content: 'x' }, { role: 'user', content: 'y' }] }, UNREADABLE],
  ['list content', { ...ok, msgs: [{ role: 'user', content: [{ type: 'text', text: 'x' }] }] }, UNREADABLE],
  ['a null message', { ...ok, msgs: [null] }, UNREADABLE],
  ['tool calls that are not text', { ...ok, msgs: [{ role: 'assistant', content: '', toolCalls: [{}] }] }, UNREADABLE],
] as const) {
  test(`share link with ${name} is refused and the saved work is kept`, async ({ page }) => {
    const before = await ownWork(page)
    await page.goto('about:blank')
    await page.goto('/app' + (await shareHash(payload)))
    await expect(page.getByRole('status')).toHaveText(message)
    expect(await page.evaluate(() => localStorage.getItem('tl:state'))).toBe(before)
  })
}

for (const hash of ['#try=', '#try=chat|', '#try=chat|%', '#try=%E0%A4%A|x', '#try=nosuch|Qwen%2FQwen3-8B']) {
  test(`preset link ${hash} says it can't be read and keeps the saved work`, async ({ page }) => {
    const before = await ownWork(page)
    await page.goto('about:blank')
    await page.goto('/app' + hash)
    await expect(page.getByRole('status')).toHaveText('That link’s sample or model can’t be read.')
    expect(await page.evaluate(() => localStorage.getItem('tl:state'))).toBe(before)
  })
}

test('a preset link works with an escaped separator', async ({ page }) => {
  await page.goto('/app#try=thinking%7CQwen%2FQwen3-8B')
  await expect(page.locator('.source .meta')).toContainText('Qwen/Qwen3-8B')
  expect((await stored(page))?.kwargs).toContain('enable_thinking')
})
