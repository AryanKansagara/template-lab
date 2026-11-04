import { lazy, Suspense, useEffect, useState } from 'react'
import Landing from './Landing.tsx'

const Lab = lazy(() => import('./Lab.tsx'))

export function go(href: string) {
  history.pushState({}, '', href)
  dispatchEvent(new PopStateEvent('popstate'))
  scrollTo(0, 0)
}

/** Client-side nav for same-origin links. */
export function Link(props: React.AnchorHTMLAttributes<HTMLAnchorElement> & { href: string }) {
  return (
    <a
      {...props}
      onClick={(e) => {
        if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return
        e.preventDefault()
        go(props.href)
      }}
    />
  )
}

function ThemeToggle() {
  const [mode, setMode] = useState(() => document.documentElement.dataset.themeMode ?? 'light')
  const next = mode === 'dark' ? 'light' : 'dark'
  return (
    <button
      className="icon-btn"
      aria-label={`Switch to ${next} theme`}
      title={`Switch to ${next} theme`}
      onClick={() => {
        document.documentElement.dataset.themeMode = next
        localStorage.setItem('themeMode', next)
        setMode(next)
      }}
    >
      {mode === 'dark' ? (
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></svg>
      ) : (
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" /></svg>
      )}
    </button>
  )
}

export function Monogram() {
  return (
    <span className="monogram" aria-hidden>
      TL
    </span>
  )
}

export default function App() {
  const [path, setPath] = useState(location.pathname)
  useEffect(() => {
    const on = () => setPath(location.pathname)
    addEventListener('popstate', on)
    return () => removeEventListener('popstate', on)
  }, [])
  const inLab = path === '/app' || path.startsWith('/app/')
  return (
    <>
      <a className="skip" href="#main">
        Skip to content
      </a>
      <header className="topbar">
        <div className="topbar-inner">
          <Link className="brand" href="/">
            <Monogram />
            <span>Template Lab</span>
          </Link>
          <nav className="nav" aria-label="Main">
            <Link href="/" aria-current={!inLab ? 'page' : undefined}>
              Overview
            </Link>
            <Link href="/app" aria-current={inLab ? 'page' : undefined}>
              Lab
            </Link>
            <a href="https://huggingface.co/docs/transformers/main/en/chat_templating" target="_blank" rel="noreferrer">
              Chat template docs
            </a>
          </nav>
          <div className="topbar-actions">
            <ThemeToggle />
            {!inLab && (
              <Link className="btn btn-dark btn-pill" href="/app">
                Open the lab
              </Link>
            )}
          </div>
        </div>
      </header>
      <main id="main">
        {inLab ? (
          <Suspense fallback={<div className="lab-loading">Loading lab…</div>}>
            <Lab />
          </Suspense>
        ) : (
          <Landing />
        )}
      </main>
      <footer className="footer">
        <div className="footer-inner">
          <span>
            Built by <a href="https://github.com/AryanKansagara">Aryan Kansagara</a>. Independent project, not affiliated with Hugging Face.
          </span>
          <span className="muted">Renders with @huggingface/jinja · tokenizes with transformers.js · runs entirely in your browser</span>
        </div>
      </footer>
    </>
  )
}
