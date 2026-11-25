import { test, expect } from './fixtures.ts'

test('overview: the hero renders what is typed and names the problem', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('tab', { name: 'Mistral-7B-Instruct-v0.1' }).click()
  await page.getByLabel('Try your own user message').fill('Why?')
  // "<s>" + "[INST] Why? [/INST]" = 3 + 19 = 22 characters.
  // This template drops the system message too, but the duplicate BOS is the error, so it is named first.
  await expect(page.locator('.demo-render')).toHaveText('<s>[INST]·​Why?·​[/INST]')
  await expect(page.locator('.demo-foot')).toContainText('22 chars')
  await expect(page.locator('.demo-foot')).toContainText('Duplicate BOS token')
})

test('overview, lab and docs are reachable, and an unknown path opens the overview', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('navigation', { name: 'Main' }).getByRole('link', { name: 'Docs' }).click()
  await expect(page.getByRole('heading', { name: 'Survey of real Hub templates' })).toBeVisible()
  await page.getByRole('navigation', { name: 'Main' }).getByRole('link', { name: 'Lab' }).click()
  await expect(page.getByRole('heading', { name: 'Conversation' })).toBeVisible()
  await page.goto('/nowhere')
  await expect(page.getByRole('heading', { level: 1 })).toContainText('See the prompt your model')
})

test('the theme is remembered', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'Switch to dark theme' }).click()
  await page.reload()
  await expect(page.locator('html')).toHaveAttribute('data-theme-mode', 'dark')
})
