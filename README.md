# Xinity Migration Examples

Migrate your app from closed-source AI APIs to your own compute—without a rewrite.

This repository contains reference adapters, configuration, and hands-on playbooks that let you **redirect API calls to your own servers running open models via Xinity's router**. Keep your app logic and SDKs largely intact while moving inference to hardware you control.  
> Goals: lower cost, greater control, data residency, and faster iteration with fine-tuning on your domain data.

---

## Why Xinity?

- **One-line traffic redirection** to your own servers, preserving your existing call patterns.
- **On-prem / hybrid**: run on hardware you already own or control.
- **Data capture for improvement**: structured request/response logging enables RLHF + PEFT fine-tuning loops.
- **Compute sovereignty**: aligns with our mission of a more compute-independent Europe and practical routes for enterprises with in-house servers. :contentReference[oaicite:1]{index=1}

---

## What’s inside
```
.
├─ adapters/
│ ├─ python/
│ │ ├─ openai_to_xinity.py
│ │ └─ anthropic_to_xinity.py
│ └─ javascript/
│ ├─ openai-to-xinity.js
│ └─ anthropic-to-xinity.js
├─ examples/
│ ├─ chat-completions/
│ └─ embeddings/
├─ playbooks/
│ ├─ openai_to_xinity.md
│ ├─ anthropic_to_xinity.md
│ └─ cohere_to_xinity.md
├─ configs/
│ ├─ model-aliases.yaml
│ └─ logging.yaml
└─ README.md
```

- **adapters/**: drop-in wrappers for popular SDKs.
- **playbooks/**: copy-paste migration steps and testing checklists.
- **configs/**: model aliasing (map closed-source names to your local models) and logging config.

> Note: This repo provides **reference code and patterns**. The Xinity router/agent endpoint URLs in examples are placeholders—point them to your deployment.

---

## Quick start

### 1) Set environment variables

```bash
export XINITY_ROUTER_URL="https://your-xinity-router.example.com"
export XINITY_API_KEY="xinity_dev_key_abc123"
# Optional: capture for training & eval
export XINITY_CAPTURE="true"

### 2) Map model aliases

`configs/model-aliases.yaml` (example)

aliases:
  gpt-4o:           "NousResearch/Hermes-3-Llama-3.1-8B"
  gpt-4o-mini:      "Qwen2.5-7B-Instruct"
  claude-3-5-sonnet:"Llama-3.1-70B-Instruct"
  text-embedding-3-large: "bge-large-en-v1.5"


### 3) Python (OpenAI-style) adapter

`adapters/python/openai_to_xinity.py`

```Python
import os
import requests

XINITY_URL = os.getenv("XINITY_ROUTER_URL").rstrip("/")
XINITY_KEY = os.getenv("XINITY_API_KEY")
CAPTURE = os.getenv("XINITY_CAPTURE", "false").lower() == "true"

def chat_completions(model, messages, temperature=0.2, top_p=1.0, **kwargs):
    payload = {
        "model": model,            # will be resolved via model-aliases.yaml on server
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "kwargs": kwargs,
        "capture": CAPTURE,
        "task_type": "chat.completions"
    }
    r = requests.post(
        f"{XINITY_URL}/v1/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {XINITY_KEY}"},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()
```

### 4) JavaScript (OpenAI-style) adapter
`adapters/javascript/openai-to-xinity.js`
```JavaScript
const XINITY_URL = (process.env.XINITY_ROUTER_URL || "").replace(/\/$/, "");
const XINITY_KEY = process.env.XINITY_API_KEY;
const CAPTURE = (process.env.XINITY_CAPTURE || "false").toLowerCase() === "true";

export async function chatCompletions({ model, messages, temperature = 0.2, top_p = 1.0, ...rest }) {
  const res = await fetch(`${XINITY_URL}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${XINITY_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model,
      messages,
      temperature,
      top_p,
      kwargs: rest,
      capture: CAPTURE,
      task_type: "chat.completions"
    })
  });
  if (!res.ok) throw new Error(`Xinity error ${res.status}: ${await res.text()}`);
  return await res.json();
}
```

`Usage:`

```JavaScript
import { chatCompletions } from "./adapters/javascript/openai-to-xinity.js";

const resp = await chatCompletions({
  model: "gpt-4o", // alias -> local
  messages: [
    { role: "system", content: "You are concise." },
    { role: "user", content: "What's the purpose of this repo?" }
  ],
});
console.log(resp.choices[0].message.content);
```

## Playbook: OpenAI → Xinity (summary)

1. **Identify features used**: chat, tools, streaming, embeddings, RAG.
2. **Model mapping**: pick local equivalents (see `configs/model-aliases.yaml`).
3. **Endpoint swap**: replace SDK calls with the adapter or set a custom base URL in your client.
4. **Functional parity**: run smoke tests (latency, cost, output adequacy).
5. **Data capture**: enable `XINITY_CAPTURE=true` to store trace + feedback for fine-tuning.
6. **Fine-tune loop**: curate positives/negatives, run PEFT job, redeploy alias to tuned model.
7. **Gradual rollout**: % traffic via router; monitor; then cut over.

See full steps in `playbooks/openai_to_xinity.md`.

---

## Security & privacy

* All requests are authenticated via bearer token.
* Optional structured logging supports redaction hooks before persistence.
* Run the router within your network boundary for data residency.

---

## Roadmap

* Tool calling / function calling parity examples
* Streaming examples (SSE/WebSocket)
* Embeddings + RAG adapters
* Eval harness (quality + cost dashboards)

---

## Contributing

PRs welcome! Please open an issue first for major changes. Make sure to update tests and playbooks.

