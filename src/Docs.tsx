import { Link } from './App.tsx'

// Survey run on 16 Nov 2026 against the public Hub with the lab's own loader, renderer and linter.
// "Tool round trip" = system, user, assistant tool_call (id abc123xyz), tool result, with one tool schema.
const SURVEY: [string, string, string, string][] = [
  ['Qwen/Qwen2.5-7B-Instruct', 'tokenizer_config.json', 'clean', 'clean'],
  ['Qwen/Qwen3-8B', 'tokenizer_config.json', 'clean', 'clean'],
  ['mistralai/Mistral-7B-Instruct-v0.3', 'tokenizer_config.json', 'double BOS; system prompt moves to the last user turn (prefix mismatch)', 'system prompt dropped when the conversation ends on a tool result'],
  ['mistralai/Mistral-Small-3.1-24B-Instruct-2503', 'chat_template.json', 'double BOS', 'raises: only user, system and assistant roles'],
  ['HuggingFaceTB/SmolLM2-1.7B-Instruct', 'tokenizer_config.json', 'clean', 'tools and tool_calls ignored'],
  ['HuggingFaceTB/SmolLM3-3B', 'chat_template.jinja', 'clean', 'tool_calls dropped (tools rendered with Python repr)'],
  ['microsoft/Phi-3.5-mini-instruct', 'tokenizer_config.json', 'clean', 'tool result dropped, tools ignored'],
  ['microsoft/phi-4', 'tokenizer_config.json', 'clean', 'tool result dropped, tools ignored'],
  ['unsloth/Llama-3.1-8B-Instruct', 'tokenizer_config.json', 'double BOS if you encode the string', 'double BOS if you encode the string'],
  ['unsloth/gemma-2-2b-it', 'tokenizer_config.json', 'raises: system role not supported', 'raises: system role not supported'],
  ['unsloth/gemma-3-4b-it', 'tokenizer_config.json', 'double BOS', 'raises: roles must alternate'],
  ['openai/gpt-oss-20b', 'chat_template.jinja', 'clean', 'clean'],
  ['deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', 'tokenizer_config.json', 'double BOS; past <think> stripped (prefix mismatch)', 'tools ignored'],
  ['deepseek-ai/DeepSeek-V3', 'tokenizer_config.json', 'double BOS', 'tools ignored'],
  ['ibm-granite/granite-3.3-8b-instruct', 'tokenizer_config.json', 'clean', 'clean'],
  ['allenai/OLMo-2-1124-7B-Instruct', 'tokenizer_config.json', 'clean', 'tool result dropped, tools ignored'],
  ['NousResearch/Hermes-3-Llama-3.1-8B', 'tokenizer_config.json (default + tool_use)', 'clean', 'default template ignores tools; pick tool_use'],
  ['moonshotai/Kimi-K2-Instruct', 'chat_template.jinja', 'clean', 'clean'],
  ['zai-org/GLM-4.5-Air', 'chat_template.jinja', 'clean', 'clean'],
  ['meta-llama/Llama-3.1-8B-Instruct', 'gated', 'gated: suggests unsloth/Llama-3.1-8B-Instruct', ''],
]

export default function Docs() {
  return (
    <div className="docs">
      <h1>Docs</h1>
      <h2>What it loads</h2>
      <ul>
        <li><b>Hub model id or URL</b> (<code>org/name</code>, or any <code>/tree/</code>, <code>/blob/</code> or <code>/resolve/</code> link). Files are resolved the way transformers does it: a standalone <code>chat_template.jinja</code> wins; otherwise <code>chat_template</code> in <code>tokenizer_config.json</code> (a string or a list of named templates); otherwise the processor's <code>chat_template.json</code>. A 404 for a file the repo doesn't have is expected.</li>
        <li><b>Pasted template</b> with your own BOS and EOS.</li>
        <li><b>Tokenizer</b>: the repo's <code>tokenizer.json</code> through transformers.js, only when you ask for token counts.</li>
      </ul>
      <h2>How rendering matches transformers</h2>
      <ul>
        <li>Templates run on <code>@huggingface/jinja</code>, the same engine transformers.js uses. <code>raise_exception</code> is reported as a template refusal, not a crash.</li>
        <li><code>x | string</code> on a dict or list prints Python's <code>repr()</code> (single quotes, <code>True</code>, <code>None</code>), as in Python transformers. SmolLM3 renders tool schemas this way.</li>
        <li><code>strftime_now</code> gives today's date, so Llama 3.x and SmolLM3 headers show the real "Today Date".</li>
      </ul>
      <h2>Survey of real Hub templates</h2>
      <p className="muted">Run on 16 Nov 2026 against huggingface.co with this app's loader and linter. Plain chat = system, user, assistant, user. Tool round trip = system, user, assistant tool_call, tool result, one tool schema. "Double BOS" means the template writes BOS and <code>add_bos_token</code> is true, so encoding the rendered string with special tokens adds a second one.</p>
      <div className="docs-table">
        <table>
          <thead><tr><th>Model</th><th>Template file</th><th>Plain chat</th><th>Tool round trip</th></tr></thead>
          <tbody>{SURVEY.map(([m, f, a, b]) => <tr key={m}><td><code>{m}</code></td><td>{f}</td><td>{a}</td><td>{b}</td></tr>)}</tbody>
        </table>
      </div>
      <h2>Test fixtures</h2>
      <p>The test suite renders real templates copied from these repos (all Apache-2.0 or the model's published licence): Qwen2.5, Qwen3, Mistral-7B-Instruct-v0.3, Llama 3.1 (Llama 3.1 Community Licence), Gemma 2 (Gemma Terms), Phi-3.5 (MIT) and SmolLM3. The Python <code>repr()</code> expectation was produced by CPython 3.</p>
      <h2>Limits</h2>
      <ul>
        <li>Gated repos cannot be fetched anonymously from a browser. Paste the template, or use the suggested ungated copy.</li>
        <li>BOS behaviour comes from <code>add_bos_token</code> until the tokenizer is loaded; then it is measured from <code>encode()</code>.</li>
        <li>Templates that call Python-only helpers beyond the ones above will show a render error.</li>
      </ul>
      <p><Link className="btn btn-dark btn-pill" href="/app">Open the lab</Link></p>
    </div>
  )
}
