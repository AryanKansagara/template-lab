/** Compress app state into a URL-safe string (deflate-raw + base64url), using native CompressionStream. */
export async function encodeState(state: unknown): Promise<string> {
  const bytes = new Uint8Array(await new Response(new Blob([JSON.stringify(state)]).stream().pipeThrough(new CompressionStream('deflate-raw'))).arrayBuffer())
  let bin = ''
  for (const b of bytes) bin += String.fromCharCode(b)
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

export async function decodeState<T>(s: string): Promise<T | null> {
  try {
    const bin = atob(s.replace(/-/g, '+').replace(/_/g, '/'))
    const bytes = Uint8Array.from(bin, (c) => c.charCodeAt(0))
    const text = await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate-raw'))).text()
    return JSON.parse(text) as T
  } catch {
    return null
  }
}
