---
title: "🤖 Jarvis — Personal AI Agent for macOS"
summary: "An autonomous AI agent that orchestrates terminal, browser, and filesystem via LLMs with a 4-tier safety system and Agent-to-Agent (A2A) protocol support."
tags:
  - Agents
  - LLM
  - macOS
  - Browser Automation
  - Python
date: 2026-02-17
links:
  - icon: github
    icon_pack: fab
    name: Source Code
    url: https://github.com/BrutalCaeser/jarvis
---

Jarvis is an autonomous AI agent that runs locally on macOS, controlling your **terminal**, **browser**, and **filesystem** to accomplish real tasks. It uses the [ReAct](https://arxiv.org/abs/2210.03629) (Reason + Act) pattern to break down complex tasks, execute shell commands, browse the web, read/write files, and interact with macOS — all through a conversational terminal interface with built-in safety guardrails.

<!--more-->

## ✨ Features

- 🔧 **Terminal Control** — Execute any zsh command with automatic output capture, timeout handling, and truncation
- 🌐 **Browser Automation** — Navigate websites, click elements, fill forms, extract text, and take screenshots via Playwright
- 📁 **File Operations** — Read and write files with blocked-path safety checks
- 🍎 **macOS Integration** — Desktop notifications, clipboard access, Spotlight search, text-to-speech
- 🛡️ **4-Tier Safety System** — Commands classified as Auto / Notify / Approve / Blocked before execution
- 📋 **Audit Logging** — Every tool invocation logged to JSONL session files
- 🔄 **Multi-Provider LLM** — Supports NVIDIA NIM, Anthropic Claude, and OpenAI GPT with automatic fallback
- 🤝 **Agent-to-Agent (A2A)** — Expose Jarvis as an A2A server or discover/message other agents

---

## 🏗️ Architecture

```text
┌──────────────────────────────────────────────────────────┐
│                      run.py (Entry Point)                │
│                      src/ui/cli.py (Rich TUI)            │
├──────────────────────────────────────────────────────────┤
│   src/agent.py — ReAct Loop                              │
│   ┌─────────────────────────────────────────────┐        │
│   │  User Input → LLM → Tool Calls → Observe   │─ loop  │
│   │  → Reflect → Respond (or repeat)            │        │
│   └─────────────────────────────────────────────┘        │
├──────────────┬───────────────┬────────────────────────────┤
│  LLM Layer   │  Tool Layer   │  Safety Layer              │
│  anthropic   │  terminal.py  │  classifier.py (4 tiers)   │
│  openai      │  filesystem   │  guardrails.py (path ACL)  │
│  nvidia      │  browser.py   │  audit.py (JSONL logging)  │
│  router.py   │  macos.py     │                            │
│              │  a2a/client   │                            │
├──────────────┴───────────────┴────────────────────────────┤
│  A2A Layer (--serve mode)                                 │
│  src/a2a/server.py — Starlette HTTP+JSON REST server      │
└──────────────────────────────────────────────────────────┘
```

---

## 🛡️ Safety System

Every shell command passes through a **4-tier classification engine** before execution:

| Tier | Action | Examples |
|---|---|---|
| **Tier 1 — Auto** ✅ | Execute immediately | `ls`, `cat`, `grep`, `git status`, `python --version` |
| **Tier 2 — Notify** 📢 | Announce, then execute | `cp`, `mkdir`, `pip install`, `git commit`, `python script.py` |
| **Tier 3 — Approve** ⚠️ | Requires explicit `y/N` | `rm`, `sudo`, `git push`, `git reset`, `docker run` |
| **Tier 4 — Blocked** 🚫 | Never executed | `rm -rf /`, fork bombs, `dd`, `curl | sh` |

**Path restrictions** deny access to `~/.ssh`, `~/.aws`, `~/.gnupg`, `/etc`, `/System`, and `/usr` by default.

---

## 🛠️ 11 Built-in Tools

| Tool | Description |
|---|---|
| `run_terminal_command` | Execute any zsh command with timeout and output truncation |
| `read_file` / `write_file` | File I/O with blocked-path enforcement |
| `browse_website` | Navigate to URL, extract visible text content |
| `browser_action` | Click, fill, scroll, screenshot, or extract links |
| `macos_notify` | Send a macOS desktop notification |
| `clipboard_read` / `clipboard_write` | Read/write clipboard via `pbpaste`/`pbcopy` |
| `spotlight_search` | Search files via `mdfind` (Spotlight) |
| `discover_agent` | Discover a remote A2A agent by fetching its Agent Card |
| `send_a2a_message` | Send a text message to a remote A2A-compatible agent |

---

## 🤝 Agent-to-Agent (A2A) Protocol

A2A lets AI agents talk to each other over the internet, like websites talk via APIs. Jarvis supports A2A in two directions:

**Jarvis as a server** — Start with `python run.py --serve` and other agents can send tasks via HTTP:
```text
POST /message:send → Jarvis thinks → runs tools → returns answer as JSON
```

**Jarvis as a client** — Ask Jarvis to discover and talk to other A2A agents in plain English:
```text
jarvis> Find out what the agent at https://ishaan.bot can do
jarvis> Ask the agent at https://ishaan.bot to write a haiku about coding
```

---

## 🔌 LLM Providers

| Provider | Models | Key |
|---|---|---|
| **NVIDIA NIM** | Kimi K2.5, Qwen 3 235B, Llama 3.3 | `NVIDIA_API_KEY` |
| **Anthropic** | Claude Sonnet, Claude Opus | `ANTHROPIC_API_KEY` |
| **OpenAI** | GPT-4o, o1 | `OPENAI_API_KEY` |

Automatic **fallback** — if the primary provider fails, Jarvis switches to the secondary provider configured in `config.yaml`.

---

## 💬 Rich Terminal UI

Interactive terminal interface with Markdown rendering, colored output, approval prompts, and command history. Slash commands: `/help`, `/clear`, `/log`, `/quit`.

---

*Built with Python · Playwright · Starlette · Rich · Multi-provider LLM · MIT License*
