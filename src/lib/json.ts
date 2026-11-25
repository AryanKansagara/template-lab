/** A whole-valued JSON float (1.0, -0.0, 1e3) or one beyond the range of a double. As a JS number it would pass for an integer, or for null. */
export class PyFloat {
  value: number
  constructor(value: number) {
    this.value = value
  }
}

const MARK = '\u0000tl:number'
const TOKEN = /"(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/g

/** A value JS numbers keep as Python has it: floats with a fraction, and integers up to 2^53. */
const plain = (src: string) => {
  const v = Number(src)
  return /[.eE]/.test(src) ? Number.isFinite(v) && !Number.isInteger(v) : Number.isSafeInteger(v)
}

/** JSON.parse, with the numbers JS would change kept as Python reads them: PyFloat, or bigint for large integers. */
export function parsePython(text: string): unknown {
  JSON.parse(text) // reports errors at their position in the original
  const marked = text.replace(TOKEN, (t) => (t[0] === '"' || plain(t) ? t : `{${JSON.stringify(MARK)}:"${t}"}`))
  return JSON.parse(marked, (_, v) => {
    const src = v && typeof v === 'object' && !Array.isArray(v) ? v[MARK] : undefined
    return typeof src !== 'string' ? v : /[.eE]/.test(src) ? new PyFloat(Number(src)) : BigInt(src)
  })
}

/** Python repr() of a float. */
export function floatRepr(v: number): string {
  if (Number.isNaN(v)) return 'nan'
  if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf'
  if (Object.is(v, -0)) return '-0.0'
  const [m, e] = v.toExponential().split('e')
  if (Number(e) >= 16 || Number(e) < -4) return `${m}e${e[0]}${e.slice(1).padStart(2, '0')}`
  return Number.isInteger(v) ? `${v}.0` : String(v)
}

/** json.dumps() of a float. */
export const floatJson = (v: number) => (Number.isNaN(v) ? 'NaN' : Number.isFinite(v) ? floatRepr(v) : v > 0 ? 'Infinity' : '-Infinity')
