import { test, expect, openLab, paste, conversation, prompt, tab, loadModel, setSwitch } from './fixtures.ts'

// Every number and string in this file was worked out on paper from the template and the vocabulary.

test('a pasted template renders the exact prompt and character count', async ({ page }) => {
  await openLab(page)
  await paste(page, '{{ bos_token }}{% for m in messages %}[{{ m.role }}] {{ m.content | trim }}{{ eos_token }}\n{% endfor %}{% if add_generation_prompt %}[assistant]{% endif %}', '<s>', '</s>')
  await conversation(page, [{ role: 'system', content: '  Be kind. ' }, { role: 'user', content: 'Hi' }])
  // <s>                         3
  // [system] Be kind.</s>\n     8 + 1 + 8 + 4 + 1 = 22
  // [user] Hi</s>\n             6 + 1 + 2 + 4 + 1 = 14
  // [assistant]                 11          total 50
  await expect.poll(() => prompt(page)).toBe('<s>[system] Be kind.</s>\n[user] Hi</s>\n[assistant]')
  await expect(page.getByText('50 chars')).toBeVisible()
  // Without the generation prompt the 11 characters of "[assistant]" go away.
  await setSwitch(page, 'add_generation_prompt', false)
  await expect(page.getByText('39 chars')).toBeVisible()
  await expect.poll(() => prompt(page)).toBe('<s>[system] Be kind.</s>\n[user] Hi</s>\n')
})

test('token ids, special count and characters per token', async ({ page }) => {
  await openLab(page)
  await loadModel(page, 'test/tiny')
  await conversation(page, [{ role: 'user', content: 'alpha beta gamma' }])
  await tab(page, 'Tokens')
  await page.getByRole('button', { name: 'Load tokenizer' }).click()
  // "<s>alpha beta gamma</s>": 3 + 16 + 4 = 23 characters, ids 1 3 4 5 2, two of them special, 23 / 5 = 4.60.
  await expect(page.locator('.stat-row')).toContainText('5 tokens')
  await expect(page.locator('.stat-row')).toContainText('2 special')
  await expect(page.locator('.stat-row')).toContainText('4.60 chars/token')
  await page.getByLabel('ids', { exact: true }).check()
  await expect(page.locator('.chip')).toHaveText(['1', '3', '4', '5', '2'])
  // A word outside the seven-word vocabulary is <unk>, id 0.
  await page.getByLabel('Message 1 content').fill('alpha zebra')
  await expect(page.locator('.chip')).toHaveText(['1', '3', '0', '2'])
})

test('compare: token delta, percentage and the diffed text', async ({ page }) => {
  await openLab(page)
  await loadModel(page, 'test/tiny')
  await conversation(page, [{ role: 'user', content: 'alpha beta gamma' }])
  await tab(page, 'Compare models')
  await page.getByLabel('Compare with model').fill('test/tiny-long')
  await page.getByRole('button', { name: 'Compare', exact: true }).click()
  await page.getByRole('button', { name: 'Exact token counts' }).click()
  // A: 1 3 4 5 2 = 5 tokens. B adds " gamma gamma ": 7 tokens. Delta +2, 2 / 5 = 40%. B is 23 + 13 = 36 characters.
  await expect(page.locator('.side-0')).toContainText('5 tokens · 23 chars')
  await expect(page.locator('.side-1')).toContainText('7 tokens · 36 chars')
  await expect(page.locator('.cmp-delta')).toContainText('+2tokens for B (40%)')
  // "</s>" and "<s>" are shared; only the text between them differs.
  await page.getByRole('tab', { name: 'Rendered' }).click()
  await page.getByLabel('whitespace').uncheck()
  await tab(page, 'Compare models')
  await expect(page.locator('.d-del')).toHaveText(['alpha beta gamma'])
  await expect(page.locator('.d-add')).toHaveText([' gamma gamma alpha beta gamma'])
})

test('double BOS is measured from the tokenizer, not assumed', async ({ page }) => {
  await openLab(page)
  await loadModel(page, 'test/tiny-bos')
  await conversation(page, [{ role: 'user', content: 'hi' }])
  const lint = page.locator('.card.lint')
  await expect(lint).toContainText('Template emits <s>')
  await tab(page, 'Tokens')
  await page.getByRole('button', { name: 'Load tokenizer' }).click()
  // encode('hi') is 1 6 here, so a rendered "<s>…" encodes to 1 1 …
  await expect(lint).toContainText('Duplicate BOS token')
  await expect(lint).toContainText('1 error')
  // The token view follows apply_chat_template (no second BOS): "<s>hi</s>" is 1 6 2.
  await page.getByLabel('ids', { exact: true }).check()
  await expect(page.locator('.chip')).toHaveText(['1', '6', '2'])
  // The same template with a tokenizer whose encode('hi') is just 6.
  await loadModel(page, 'test/tiny')
  await expect(lint).toContainText('Single BOS')
  await expect(lint).not.toContainText('Duplicate BOS token')
})
