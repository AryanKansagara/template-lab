import { useEffect, useState } from 'react'
import { parseModelRef } from './lib/hub.ts'

/** Minimal surface we use from transformers.js' PreTrainedTokenizer. */
export interface Tok {
  encode(text: string, opts?: { add_special_tokens?: boolean }): number[]
  decode(ids: number[], opts?: { skip_special_tokens?: boolean }): string
}

const cache = new Map<string, Promise<Tok>>()

/** `raw` is whatever was loaded: an id or a Hub URL, whose revision the tokenizer comes from too. */
export function loadTokenizer(raw: string): Promise<Tok> {
  let p = cache.get(raw)
  if (!p) {
    p = import('@huggingface/transformers').then(async ({ AutoTokenizer, env }) => {
      // SPA rewrites would answer /models/... with index.html; only ever hit the Hub.
      env.allowLocalModels = false
      const { id, revision } = parseModelRef(raw) ?? { id: raw, revision: 'main' }
      return (await AutoTokenizer.from_pretrained(id, { revision })) as unknown as Tok
    })
    p.catch(() => cache.delete(raw))
    cache.set(raw, p)
  }
  return p
}

/** Does encode() prepend a BOS on its own? (add_bos_token or a TemplateProcessing post-processor) */
export function tokenizerAddsBos(tok: Tok, bos?: string): boolean | null {
  if (!bos) return null
  const bosIds = tok.encode(bos, { add_special_tokens: false })
  if (bosIds.length !== 1) return null
  return tok.encode('hi')[0] === bosIds[0]
}

export type TokState = { status: 'off' } | { status: 'loading' } | { status: 'ready'; tok: Tok } | { status: 'error'; message: string }

export function useTokenizer(id: string | null, enabled: boolean): TokState {
  const [state, setState] = useState<TokState>({ status: 'off' })
  useEffect(() => {
    if (!enabled || !id) return setState({ status: 'off' })
    let live = true
    setState({ status: 'loading' })
    loadTokenizer(id).then(
      (tok) => live && setState({ status: 'ready', tok }),
      (e: Error) => {
        if (!live) return
        const msg = /401|403|gated|Unauthorized/i.test(e.message)
          ? `${id} is gated, so its tokenizer can't be downloaded anonymously. Use an ungated mirror for token counts.`
          : /404|not found|Could not locate/i.test(e.message)
            ? `${id} has no tokenizer.json transformers.js can load.`
            : e.message
        setState({ status: 'error', message: msg })
      },
    )
    return () => {
      live = false
    }
  }, [id, enabled])
  return state
}
