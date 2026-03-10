---
title: "🎯 Hunter — Autonomous Job Application Agent"
summary: "An LLM-driven browser automation agent that discovers internship roles, fills applications deterministically, and runs 24/7 on macOS."
tags:
  - Agents
  - LLM
  - Browser Automation
  - Python
  - Productivity
date: 2026-02-24
links:
  - icon: github
    icon_pack: fab
    name: Source Code
    url: https://github.com/BrutalCaeser/Job_Hunter
---

Hunter discovers internship roles, fills applications via **LLM-driven browser automation**, and runs **24/7 on macOS** with zero babysitting. It uses an adaptive deterministic form-filling pipeline with LLM fallback — most fields are matched from a YAML profile and cached answers (~$0/field), with a single Claude Haiku call for novel custom questions (~$0.010/application).

<!--more-->

## Architecture

```text
Scheduler (daemon.py / launchd)
    │
    ├─ Every 3 hrs ─▶ Discovery Engine ─▶ SQLite (data/hunter.db)
    │
    ├─ Every 4 hrs ─▶ Application Engine ◀─ SQLite
    │                     │
    │                     ├─ Account Manager (create account, click verify email)
    │                     ├─ Form Filler    (deterministic + LLM fallback)
    │                     │     ├─ Scan fields → rule-match from profile.yaml
    │                     │     ├─ Fuzzy dropdown matching + synonym type-ahead
    │                     │     ├─ Answer cache (SQLite) for repeat patterns
    │                     │     └─ LLM batch call for novel custom questions
    │                     └─ Verifier       (coverage check + critical field match)
    │
    └─ Daily 8pm ──▶ Reporter (rich table in terminal + macOS notification)
```

---

## Key Features

### 🔍 Discovery Engine
- **SpeedyApply** — Scrapes curated GitHub markdown job lists via pure HTTP (no browser needed)
- **Career Pages** — Uses Browser Use + LLM to navigate company career pages
- **Relevance Scoring** — 0.0–1.0 scoring with configurable keywords and thresholds

### 📝 Adaptive Form Filling
- **Deterministic first** — Profile YAML → field rules → fuzzy dropdown matching
- **Learning layer** — Learned dropdown mappings and cached answers stored in SQLite
- **LLM fallback** — Claude Haiku for genuinely novel questions only
- **Verification pipeline** — Every field is read back and compared before submission

### 🔐 Safety by Default

| Setting | Default | Purpose |
|---|---|---|
| AUTO_SUBMIT | `False` | Manual review before every submission |
| headless | `False` | Watch every browser action in real time |
| max_applications | 3/run | Prevent runaway batch applications |

### 🏢 ATS Support
- **Greenhouse, Lever, Ashby** — Full deterministic filling
- **iCIMS** — Automatic account creation + IMAP email verification
- **Workday** — Dedicated scanner + page filler (manual sign-in for bot detection)

---

## Multi-Provider LLM

| Provider | Use Case | Cost |
|---|---|---|
| **Claude Haiku** | Form filling (precision critical) | ~$0.010/app |
| **NVIDIA NIM** | Discovery scraping (free tier) | $0 |
| **OpenAI** | Alternative provider | Variable |

After the learning layer stabilizes, repeat ATS patterns cost ~$0.007/app. Monthly total for 50–100 applications: **~$1–2**.

---

## Database & Learning

SQLite stores everything locally — jobs, application fields, accounts, learned dropdown mappings, cached answers, and daily reports. All personal data stays on your machine (profile, cookies, DB all in `.gitignore`).

Job lifecycle: `discovered` → `queued` → `applied` / `failed` / `needs_review` / `skipped`

---

## Daemon Modes

Two deployment options:
- **launchd** — Built-in macOS scheduler (recommended)
- **OpenClaw** — With Telegram integration for remote status pings

Discovery runs every 3 hours, applications every 4 hours, daily report at 8pm.

---

*Built with Python · Browser Use + Playwright · Claude Haiku · SQLite · macOS native · MIT License*
