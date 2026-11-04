# Template Lab

A debugger for Hugging Face chat templates: see the exact prompt a model gets, and what the template silently did to your conversation.

**Live:** https://template-lab-mu.vercel.app

## The pain

`chat_template` is a Jinja program hidden in `tokenizer_config.json`. When it goes wrong nothing crashes: the system prompt disappears, tools never reach the model, BOS gets added twice, and quality just drops. Today you find out by adding print statements around `apply_chat_template`.

## Features

- Load any public Hub model by id or URL, from `tokenizer_config.json`, `chat_template.jinja` or `chat_template.json`, including named templates. Or paste a template.
- Gated repos are explained, with a verified ungated copy to try.
- Conversation editor: system, user, assistant and tool messages, tool calls, tool schemas, `add_generation_prompt`, extra kwargs. Renders live with `@huggingface/jinja`.
- Output with special tokens highlighted and whitespace drawn.
- Real tokenization with `@huggingface/transformers` (lazy-loaded): ids, counts, colored chips.
- Linter: double BOS, ignored generation prompt, dropped system or tool messages, ignored tools, dropped tool calls, `raise_exception` paths, trailing whitespace, inference/training prefix mismatch, unused kwargs, missing template.
- Compare one conversation across two models, with a prompt diff and token delta.
- Shareable URLs: the whole setup is compressed into the hash. Everything runs in the browser.

## Run locally

```sh
npm install
npm run dev
npm test
npm run build
```
