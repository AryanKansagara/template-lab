import { test, expect, openLab, loadModel, prompt, tab, issue, conversation, setSwitch, stored, wordTokenizer } from './fixtures.ts'

const meta = '.source .meta'

test('opens on the default Hub model and renders its template', async ({ page }) => {
  await openLab(page)
  await expect(page.locator(meta)).toContainText('tokenizer_config.json')
  await conversation(page, [{ role: 'system', content: 'You are terse.' }, { role: 'user', content: 'Hi' }])
  await expect.poll(() => prompt(page)).toBe('<|im_start|>system\nYou are terse.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n')
})

test('loads by id, by URL and from a quick pick', async ({ page }) => {
  await openLab(page)
  await loadModel(page, 'https://huggingface.co/test/jinja-wins/blob/main/chat_template.jinja')
  await expect(page.locator(meta)).toContainText('test/jinja-wins')
  await expect(page.locator(meta)).toContainText('chat_template.jinja')
  // chat_template.jinja beats the template in tokenizer_config.json, whose special tokens still count.
  await expect(page.locator(meta)).toContainText('eos </s>')
  await conversation(page, [{ role: 'user', content: 'hey' }])
  await expect.poll(() => prompt(page)).toBe('JINJA:hey')
  await loadModel(page, 'test/processor')
  await expect(page.locator(meta)).toContainText('chat_template.json')
  await expect.poll(() => prompt(page)).toBe('PROC:hey')
  await page.getByRole('button', { name: 'Phi-3.5-mini-instruct' }).click()
  await expect(page.locator(meta)).toContainText('microsoft/Phi-3.5-mini-instruct')
  await expect.poll(() => prompt(page)).toContain('<|user|>\nhey<|end|>')
})

test('named templates can be picked', async ({ page }) => {
  await openLab(page)
  await loadModel(page, 'test/named')
  await expect(page.locator(meta)).toContainText('test/named')
  await expect.poll(() => prompt(page)).toBe('DEFAULT')
  await setSwitch(page, 'tools', true)
  await page.getByLabel('Named template').selectOption('tool_use')
  await expect.poll(() => prompt(page)).toBe('TOOLS:1')
})

test('an explicit choice of named template wins over the automatic one and survives a reload', async ({ page }) => {
  await openLab(page)
  await loadModel(page, 'test/named')
  await setSwitch(page, 'tools', true)
  await expect(page.getByLabel('Named template').locator('option').first()).toHaveText('automatic (tool_use)')
  await page.getByLabel('Named template').selectOption('default')
  await expect.poll(() => prompt(page)).toBe('DEFAULT')
  await page.reload()
  await expect.poll(() => prompt(page)).toBe('DEFAULT')
  await page.getByLabel('Named template').selectOption('')
  await expect.poll(() => prompt(page)).toBe('TOOLS:1')
})

test('named templates without a default raise as in transformers until one is picked', async ({ page }) => {
  await openLab(page)
  await loadModel(page, 'test/no-default')
  await expect(issue(page, 'No default template')).toContainText("Available template names are ['rag', 'tool_use']")
  await tab(page, 'Rendered')
  await expect(page.locator('.render-error')).toContainText('ValueError')
  await expect(page.getByRole('button', { name: 'Copy', exact: true })).toHaveCount(0)
  await page.getByLabel('Named template').selectOption('rag')
  await expect.poll(() => prompt(page)).toBe('RAG')
  await page.getByLabel('Named template').selectOption('')
  await setSwitch(page, 'tools', true)
  await expect.poll(() => prompt(page)).toBe('TOOLS:1')
})

test('compare picks the named template of model B the same way', async ({ page }) => {
  await openLab(page)
  await tab(page, 'Compare models')
  await page.getByLabel('Compare with model').fill('test/no-default')
  await page.getByRole('button', { name: 'Compare', exact: true }).click()
  await expect(page.locator('.cmp-delta')).toHaveText('No delta: a model has no default template')
  await setSwitch(page, 'tools', true)
  await expect(page.locator('.cmp-col').nth(1).locator('.rendered')).toHaveText('TOOLS:1')
  await expect(page.locator('.side-1')).toContainText('test/no-default · tool_use')
})

test('a saved pick from before the automatic choice is kept', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('tl:state', JSON.stringify({ model: 'test/named', tplName: 'tool_use' })))
  await openLab(page)
  await expect(page.getByLabel('Named template')).toHaveValue('tool_use')
  await tab(page, 'Template')
  await expect(page.getByLabel('Chat template source')).toHaveValue('TOOLS:{{ tools | length }}')
})

test('a repo without a template is an error, not a clean prompt', async ({ page }) => {
  await openLab(page)
  await loadModel(page, 'test/base-model')
  await expect(page.locator(meta)).toContainText('no chat_template')
  await expect(issue(page, 'No chat template')).toHaveClass(/sev-error/)
  await expect(page.locator('.card.lint .counts')).not.toContainText('clean')
})

test('gated model: explained, with an ungated copy that loads', async ({ page }) => {
  await openLab(page)
  await page.getByRole('button', { name: 'Llama-3.1 (gated)' }).click()
  const card = page.locator('.hub-error')
  await expect(card).toContainText('Gated model')
  await expect(page.getByRole('button', { name: 'Copy', exact: true })).toHaveCount(0)
  await card.getByRole('button', { name: 'unsloth/Llama-3.1-8B-Instruct' }).click()
  await expect(page.locator(meta)).toContainText('unsloth/Llama-3.1-8B-Instruct')
})

for (const [id, title, text] of [
  ['test/does-not-exist', 'Model not found', 'No public model called test/does-not-exist'],
  ['not a model id', 'Not a model id', 'Model ids look like org/name'],
  ['test/broken', 'Network problem', 'huggingface.co answered 503'],
  ['test/empty-config', 'Not a model id', 'tokenizer_config.json is not valid JSON'],
  ['test/truncated-config', 'Not a model id', 'tokenizer_config.json is not valid JSON'],
  ['test/html-config', 'Not a model id', 'tokenizer_config.json is not valid JSON'],
  ['test/null-config', 'Not a model id', "tokenizer_config.json can't be read: it does not hold a JSON object"],
  ['test/list-config', 'Not a model id', "tokenizer_config.json can't be read: it does not hold a JSON object"],
  ['test/null-processor', 'Not a model id', "chat_template.json can't be read: it does not hold a JSON object"],
  ['test/empty-processor', 'Not a model id', 'chat_template.json is not valid JSON'],
]) {
  test(`unusable model "${id}" says why and renders nothing`, async ({ page }) => {
    await openLab(page)
    await expect(page.locator(meta)).toContainText('Qwen')
    await loadModel(page, id)
    await expect(page.locator('.hub-error')).toContainText(title)
    await expect(page.locator('.hub-error')).toContainText(text)
    await expect(page.getByRole('button', { name: 'Copy', exact: true })).toHaveCount(0)
    await expect(page.locator('.card.lint .counts')).toHaveCount(0)
    await tab(page, 'Rendered')
    await expect(page.locator('.rendered')).toHaveCount(0)
  })
}

test('a missing tokenizer.json is reported and no count is shown', async ({ page }) => {
  await openLab(page)
  await loadModel(page, 'test/base-model')
  await tab(page, 'Tokens')
  await page.getByRole('button', { name: 'Load tokenizer' }).click()
  await expect(page.locator('.pane .err-text')).toContainText('test/base-model')
  await expect(page.locator('.stat-row')).toHaveCount(0)
})

test('editing a Hub template, resetting it, and the edit surviving a reload', async ({ page }) => {
  await openLab(page)
  await tab(page, 'Template')
  await expect(page.getByLabel('Chat template source')).toHaveValue(/im_start/)
  await page.getByLabel('Chat template source').fill('EDITED {{ messages | length }}')
  await conversation(page, [{ role: 'user', content: 'a' }, { role: 'assistant', content: 'b' }, { role: 'user', content: 'c' }])
  await expect.poll(() => prompt(page)).toBe('EDITED 3')
  expect((await stored(page))?.override).toBe('EDITED {{ messages | length }}')
  await page.reload()
  await expect(page.locator(meta)).toContainText('edited')
  await expect.poll(() => prompt(page)).toBe('EDITED 3')
  await page.getByRole('button', { name: 'edited · reset' }).click()
  await expect.poll(() => prompt(page)).toContain('<|im_start|>user\na<|im_end|>')
})

test('a PR revision is used for the template, the config and the tokenizer, and is shown', async ({ page, hub }) => {
  hub.files['test/tiny@refs/pr/1/tokenizer_config.json'] = { body: { tokenizer_class: 'PreTrainedTokenizerFast', bos_token: '<s>', eos_token: '</s>', unk_token: '<unk>', chat_template: 'alpha {{ messages[0].content }}' } }
  // Same words, other ids, so the tokenizer's revision shows.
  hub.files['test/tiny@refs/pr/1/tokenizer.json'] = { body: { ...wordTokenizer, model: { ...wordTokenizer.model, vocab: { '<unk>': 0, '<s>': 1, '</s>': 2, alpha: 30, hi: 60 } } } }
  await openLab(page)
  await loadModel(page, 'https://huggingface.co/test/tiny/blob/refs%2Fpr%2F1/tokenizer_config.json?library=transformers#L3')
  await expect(page.locator(meta)).toContainText('revision refs/pr/1')
  await conversation(page, [{ role: 'user', content: 'hi' }])
  await expect.poll(() => prompt(page)).toBe('alpha hi')
  await tab(page, 'Tokens')
  await page.getByRole('button', { name: 'Load tokenizer' }).click()
  await page.getByLabel('ids', { exact: true }).check()
  await expect(page.locator('.chip')).toHaveText(['30', '60'])
  await expect(page.locator('.pane')).toContainText('test/tiny@refs/pr/1')
  // transformers.js asks whether main has a tokenizer_config.json before loading any revision; nothing else comes from main.
  expect(hub.seen.filter((p) => p.includes('/test/tiny/') && !p.includes('/resolve/refs%2Fpr%2F1/'))).toEqual(['/test/tiny/resolve/main/tokenizer_config.json'])
  // Back on main the tag goes away and main's files are used.
  await loadModel(page, 'https://huggingface.co/test/tiny/tree/main')
  await expect(page.locator(meta)).not.toContainText('revision')
  await expect(page.locator('.chip')).toHaveText(['1', '6', '2'])
})

test('a revision that does not exist is an error, not main', async ({ page }) => {
  await openLab(page)
  await loadModel(page, 'https://huggingface.co/test/jinja-wins/tree/v9')
  await expect(page.locator('.hub-error')).toContainText('test/jinja-wins has no revision called v9.')
  await expect(page.getByRole('button', { name: 'Copy', exact: true })).toHaveCount(0)
})
