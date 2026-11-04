import type { Role } from './lib/template.ts'

export interface UIMsg {
  role: Role
  content: string
  /** assistant only: JSON array of tool calls */
  toolCalls?: string
  /** tool only */
  toolCallId?: string
}

export interface Sample {
  id: string
  label: string
  blurb: string
  model?: string
  msgs: UIMsg[]
  tools?: string
  kwargs?: string
  agp: boolean
}

export const WEATHER_TOOLS = JSON.stringify(
  [
    {
      type: 'function',
      function: {
        name: 'get_weather',
        description: 'Get the current weather for a city',
        parameters: { type: 'object', properties: { city: { type: 'string', description: 'City name' } }, required: ['city'] },
      },
    },
  ],
  null,
  2,
)

export const SAMPLES: Sample[] = [
  {
    id: 'chat',
    label: 'Multi-turn chat',
    blurb: 'System prompt + two turns. Try it on Gemma 2 or Phi-3.5.',
    agp: true,
    msgs: [
      { role: 'system', content: 'You are a concise assistant. Answer in one sentence.' },
      { role: 'user', content: 'What is a chat template?' },
      { role: 'assistant', content: 'A Jinja program that turns a list of messages into the exact string a model was trained on.' },
      { role: 'user', content: 'Why does it matter?' },
    ],
  },
  {
    id: 'tools',
    label: 'Tool-call round trip',
    blurb: 'Tools, an assistant tool_call and a tool result. Many templates silently drop one of them.',
    agp: true,
    tools: WEATHER_TOOLS,
    msgs: [
      { role: 'system', content: 'You can call tools.' },
      { role: 'user', content: "What's the weather in Paris?" },
      {
        role: 'assistant',
        content: '',
        toolCalls: JSON.stringify([{ id: 'abc123xyz', type: 'function', function: { name: 'get_weather', arguments: { city: 'Paris' } } }], null, 2),
      },
      { role: 'tool', content: '{"temp_c": 21, "sky": "clear"}', toolCallId: 'abc123xyz' },
    ],
  },
  {
    id: 'alternation',
    label: 'Broken alternation',
    blurb: 'Two user messages in a row: Mistral and Gemma raise, ChatML models shrug.',
    model: 'mistralai/Mistral-7B-Instruct-v0.3',
    agp: true,
    msgs: [
      { role: 'user', content: 'Here is my document:' },
      { role: 'user', content: 'Summarize it in three bullets.' },
    ],
  },
  {
    id: 'thinking',
    label: 'Reasoning off (enable_thinking)',
    blurb: 'Qwen3 reads enable_thinking; most other templates ignore it.',
    model: 'Qwen/Qwen3-8B',
    agp: true,
    kwargs: '{\n  "enable_thinking": false\n}',
    msgs: [
      { role: 'user', content: 'Is 9.11 bigger than 9.9?' },
    ],
  },
]

/** Small real templates used by the landing hero (rendered live, same engine as the lab). */
export const HERO_TEMPLATES = [
  {
    model: 'Qwen/Qwen2.5-7B-Instruct',
    bos: '',
    template:
      "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
  },
  {
    model: 'meta-llama/Meta-Llama-3-8B-Instruct',
    bos: '<|begin_of_text|>',
    template:
      "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}",
  },
  {
    model: 'mistralai/Mistral-7B-Instruct-v0.1',
    bos: '<s>',
    template:
      "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}",
  },
]
