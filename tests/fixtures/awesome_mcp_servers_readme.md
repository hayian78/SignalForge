[![ไทย](https://img.shields.io/badge/Thai-Click-blue)](README-th.md)
[![English](https://img.shields.io/badge/English-Click-yellow)](README.md)
[![繁體中文](https://img.shields.io/badge/繁體中文-點擊查看-orange)](README-zh_TW.md)
[![简体中文](https://img.shields.io/badge/简体中文-点击查看-orange)](README-zh.md)
[![日本語](https://img.shields.io/badge/日本語-クリック-青)](README-ja.md)
[![한국어](https://img.shields.io/badge/한국어-클릭-yellow)](README-ko.md)
[![Português Brasileiro](https://img.shields.io/badge/Português_Brasileiro-Clique-green)](README-pt_BR.md)
[![Discord](https://img.shields.io/discord/1312302100125843476?logo=discord&label=discord)](https://glama.ai/mcp/discord)
[![Subreddit subscribers](https://img.shields.io/reddit/subreddit-subscribers/mcp?style=flat&logo=reddit&label=subreddit)](https://www.reddit.com/r/mcp/)

> [!IMPORTANT]
> [Awesome MCP Servers](https://glama.ai/mcp/servers) web directory.

A curated list of awesome Model Context Protocol (MCP) servers.

* [What is MCP?](#what-is-mcp)
* [Clients](#clients)
* [Tutorials](#tutorials)
* [Community](#community)
* [Legend](#legend)
* [Server Implementations](#server-implementations) 
* [Frameworks](#frameworks)
* [Tips & Tricks](#tips-and-tricks)

## What is MCP?

[MCP](https://modelcontextprotocol.io/) is an open protocol that enables AI models to securely interact with local and remote resources through standardized server implementations. This list focuses on production-ready and experimental MCP servers that extend AI capabilities through file access, database connections, API integrations, and other contextual services.

## Clients

Checkout [awesome-mcp-clients](https://github.com/punkpeye/awesome-mcp-clients/) and [glama.ai/mcp/clients](https://glama.ai/mcp/clients).

## Tutorials

* [Tool Definition Quality Score (TDQS)](https://github.com/glama-ai/tool-definition-quality-score)
* [Model Context Protocol (MCP) Quickstart](https://glama.ai/blog/2024-11-25-model-context-protocol-quickstart)
* [Setup Claude Desktop App to Use a SQLite Database](https://youtu.be/wxCCzo9dGj0)

## Community

* [r/mcp Reddit](https://www.reddit.com/r/mcp)
* [Discord Server](https://glama.ai/mcp/discord)

## Legend

* 🎖️ – official implementation
* programming language
  * 🐍 – Python codebase
  * 📇 – TypeScript (or JavaScript) codebase
  * 🏎️ – Go codebase
  * 🦀 – Rust codebase
  * #️⃣ - C# Codebase
  * ☕ - Java codebase
  * 🌊 – C/C++ codebase
  * 💎 - Ruby codebase

* scope
  * ☁️ - Cloud Service
  * 🏠 - Local Service
  * 📟 - Embedded Systems

## Servers

- [Correctover/mcp-server](https://github.com/Correctover/mcp-server) [![Correctover MCP server](https://glama.ai/mcp/servers/Correctover/mcp-server/badges/score.svg)](https://glama.ai/mcp/servers/Correctover/mcp-server) 📇 ☁️ 🏠 🍎 🪟 🐧 - Contract validation and self-healing failover for LLM APIs. 6-dimension verification (structure, schema, latency, cost, identity, integrity) in 22μs P50. 87 self-healing rules with MAPE-K autonomic loop. BYOK direct connect to 9 providers (OpenAI, Anthropic, DeepSeek, Moonshot, Zhipu AI, Qwen, SiliconFlow, Groq, Together AI). L3 failover in 949ms E2E. Install: `npx -y correctover-mcp-server`.
- [daedalusdevelopmentgroup/ddg-agent-payable-services](https://github.com/daedalusdevelopmentgroup/ddg-agent-payable-services) [![daedalusdevelopmentgroup/ddg-agent-payable-services MCP server](https://glama.ai/mcp/servers/daedalusdevelopmentgroup/ddg-agent-payable-services/badges/score.svg)](https://glama.ai/mcp/servers/daedalusdevelopmentgroup/ddg-agent-payable-services) 🐍 ☁️ - Pay-per-call x402 gateway: one MCP server for 90+ agent tools (utilities, DNS/WHOIS, blockchain RPC, market data, prediction markets, DEX data, security audits) plus an OpenAI-compatible LLM gateway. USDC on Base, free-trial calls per agent. `pip install ddg-agent-services-mcp` or remote `https://mcp.daedalusdevelopmentgroup.com/mcp`.
- [forgemeshlabs/coinopai-mcp](https://github.com/forgemeshlabs/coinopai-mcp) [![forgemeshlabs/coinopai-mcp MCP server](https://glama.ai/mcp/servers/forgemeshlabs/coinopai-mcp/badges/score.svg)](https://glama.ai/mcp/servers/forgemeshlabs/coinopai-mcp) 📇 - Local stdio MCP server for x402-powered paid crypto intelligence: preflight checks, trade decisions with `decision_id`, later audit against real prices, risk state, signal history, and agent automation search over USDC micropayments on Base.
- [forgemeshlabs/anomaly-mcp](https://github.com/forgemeshlabs/anomaly-mcp) [![forgemeshlabs/anomaly-mcp MCP server](https://glama.ai/mcp/servers/forgemeshlabs/anomaly-mcp/badges/score.svg)](https://glama.ai/mcp/servers/forgemeshlabs/anomaly-mcp) 📇 ☁️ - Real-time anomaly detection powered by NASA-derived sequence mining across blockchain, mempool, stablecoin depeg, aviation, and GitHub signals via x402 USDC micropayments on Base. `npx -y @forgemeshlabs/anomaly-mcp`
- [2s-io/sdk](https://github.com/2s-io/sdk) [![2s-io/sdk MCP server](https://glama.ai/mcp/servers/2s-io/sdk/badges/score.svg)](https://glama.ai/mcp/servers/2s-io/sdk) 📇 ☁️ 🍎 🪟 🐧 - Unified API for AI agents — 180+ tools across geocoding, weather (NWS), climate stations (NOAA), earthquakes (USGS), tides (NOAA), points of interest (OpenStreetMap), patents (USPTO ODP), US case law (CourtListener / Free Law Project), Federal Register, Wikipedia, scientific papers (arXiv / PubMed / Semantic Scholar), AI summarize / translate / extract / screenshot / image-describe, image compression, DNS / WHOIS, crypto address-validate + EVM gas oracle, OFAC sanctions screening, US Census ACS demographics, airport / ZIP lookup. Sub-cent to a few cents per call in USDC on Base via x402 — no API keys, no signup. `npx -y @2sio/mcp`
- [8randonpickart5/alderpost-mcp](https://github.com/8randonpickart5/alderpost-mcp) [![alderpost-mcp MCP server](https://glama.ai/mcp/servers/8randonpickart5/alderpost-mcp/badges/score.svg)](https://glama.ai/mcp/servers/8randonpickart5/alderpost-mcp) 📇 ☁️ - 8 bundled intelligence endpoints (security, company, threat, compliance, sales, sports, property, health) via x402 micropayments on Base.
- [GTCC777/pulsenetwork-mcp](https://github.com/GTCC777/pulsenetwork-mcp) [![GTCC777/pulsenetwork-mcp MCP server](https://glama.ai/mcp/servers/GTCC777/pulsenetwork-mcp/badges/score.svg)](https://glama.ai/mcp/servers/GTCC777/pulsenetwork-mcp) 📇 - One MCP server exposing 66 specialized intelligence APIs (660+ endpoints) — finance, crypto, legal, immigration, healthcare cost, real estate, tax, climate, sports, science, and more — each an x402 pay-per-call tool in USDC on Base and Solana. Includes a `discover` meta-tool over the whole network and a cross-vertical referral graph. No API keys. `npx mcp-pulsenetwork`
- [mcpqueen/mcpqueen](https://github.com/mcpqueen/mcpqueen) [![mcpqueen/mcpqueen MCP server](https://glama.ai/mcp/servers/mcpqueen/mcpqueen/badges/score.svg)](https://glama.ai/mcp/servers/mcpqueen/mcpqueen) 🎖️ 📇 ☁️ - The graded MCP registry: live-probes every remote server in the official registry (initialize, tools/list, schema quality, latency, provenance) and publishes evidence-backed grades — searchable by agents via its own MCP endpoint at [mcpqueen.com](https://mcpqueen.com).
- [szp2005/llm-prices-cn](https://github.com/szp2005/llm-prices-cn) [![szp2005/llm-prices-cn MCP server](https://glama.ai/mcp/servers/szp2005/llm-prices-cn/badges/score.svg)](https://glama.ai/mcp/servers/szp2005/llm-prices-cn) 🐍 ☁️ - Daily-verified LLM API pricing dataset (44+ models, CN & global) with a hosted MCP server for live price queries and token cost estimation.
- [tadas-github/a2asearch-mcp](https://github.com/tadas-github/a2asearch-mcp) [![tadas-github/a2asearch-mcp MCP server](https://glama.ai/mcp/servers/tadas-github/a2asearch-mcp/badges/score.svg)](https://glama.ai/mcp/servers/tadas-github/a2asearch-mcp) 📇 ☁️ - MCP server to search 4,800+ MCP servers, AI agents, CLI tools and agent skills. Install: `npx -y a2asearch-mcp`. Ask Claude: "Find MCP servers for database access". Free, no auth required.
- [agentbodegastore/agentbodega](https://github.com/agentbodegastore/agentbodega) [![agentbodega MCP server](https://glama.ai/mcp/servers/agentbodegastore/agentbodega/badges/score.svg)](https://glama.ai/mcp/servers/agentbodegastore/agentbodega) 📇 ☁️ - AgentBodega MCP gives agents a live catalog of 65 x402-payable HTTP tools across search, public data, social media, status checks, media conversion, and agent-readiness checks. No API keys; install with `npx -y @agentbodega/mcp`.
- [elisymlabs/elisym](https://github.com/elisymlabs/elisym) [![elisymlabs/elisym MCP server](https://glama.ai/mcp/servers/elisymlabs/elisym/badges/score.svg)](https://glama.ai/mcp/servers/elisymlabs/elisym) 📇 ☁️ 🍎 🪟 🐧 - AI agent discovery and marketplace on Nostr with Solana payments (SOL, USDC). NIP-89 discovery, NIP-90 jobs, NIP-44 v2 encryption, on-chain micropayments.
- [1mcp/agent](https://github.com/1mcp-app/agent) 📇 ☁️ 🏠 🍎 🪟 🐧 - A unified Model Context Protocol server implementation that aggregates multiple MCP servers into one.
- [Aganium/agenium](https://github.com/Aganium/agenium) 📇 ☁️ 🍎 🪟 🐧 - Bridge any MCP server to the agent:// network — DNS-like identity, discovery, and trust for AI agents. Makes your tools discoverable and callable by other agents via `agent://` URIs with mTLS, trust scores, and capability search.
- [espadaw/Agent47](https://github.com/espadaw/Agent47) 📇 ☁️ - Unified job aggregator for AI agents across 9+ platforms (x402, RentAHuman, Virtuals, etc).
- [AgentHotspot](https://github.com/AgentHotspot/agenthotspot-mcp) 🐍 ☁️ 🏠 🍎 🪟 🐧 - Search, integrate and monetize MCP connectors on the AgentHotspot MCP marketplace
- [ariekogan/ateam-mcp](https://github.com/ariekogan/ateam-mcp) 📇 ☁️ 🏠 🍎 🪟 🐧 - Build, validate, and deploy multi-agent AI solutions on the ADAS platform. Design skills with tools, manage solution lifecycle, and connect from any AI environment via stdio or HTTP.
- [askbudi/roundtable](https://github.com/askbudi/roundtable) 📇 ☁️ 🏠 🍎 🪟 🐧 - Meta-MCP server that unifies multiple AI coding assistants (Codex, Claude Code, Cursor, Gemini) through intelligent auto-discovery and standardized MCP interface, providing zero-configuration access to the entire AI coding ecosystem.
- [blockrunai/blockrun-mcp](https://github.com/blockrunai/blockrun-mcp) 📇 ☁️ 🍎 🪟 🐧 - Access 30+ AI models (GPT-5, Claude, Gemini, Grok, DeepSeek) without API keys. Pay-per-use via x402 micropayments with USDC on Base.
- [Data-Everything/mcp-server-templates](https://github.com/Data-Everything/mcp-server-templates) 📇 🏠 🍎 🪟 🐧 - One server. All tools. A unified MCP platform that connects many apps, tools, and services behind one powerful interface—ideal for local devs or production agents.
- [duaraghav8/MCPJungle](https://github.com/duaraghav8/MCPJungle) 🏎️ 🏠 - Self-hosted MCP Server registry for enterprise AI Agents
- [edgarriba/prolink](https://github.com/edgarriba/prolink) 🐍 ☁️ 🏠 🍎 🪟 🐧 - Agent-to-agent marketplace middleware — MCP-native discovery, negotiation, and transaction between AI agents
- [glenngillen/mcpmcp-server](https://github.com/glenngillen/mcpmcp-server) ☁️ 📇 🍎 🪟 🐧 - A list of MCP servers so you can ask your client which servers you can use to improve your daily workflow.
- [hamflx/imagen3-mcp](https://github.com/hamflx/imagen3-mcp) 📇 🏠 🪟 🍎 🐧 - A powerful image generation tool using Google's Imagen 3.0 API through MCP. Generate high-quality images from text prompts with advanced photography, artistic, and photorealistic controls.
- [hashgraph-online/hashnet-mcp-js](https://github.com/hashgraph-online/hashnet-mcp-js) 📇 ☁️ 🍎 🪟 🐧 - MCP server for the Registry Broker. Discover, register, and chat with AI agents on the Hashgraph network.
- [isaac-levine/forage](https://github.com/isaac-levine/forage) 📇 🏠 🍎 🪟 🐧 - Self-improving tool discovery for AI agents. Searches registries, installs MCP servers as subprocesses, and persists tool knowledge across sessions — no restarts needed.
- [jaspertvdm/mcp-server-gemini-bridge](https://github.com/jaspertvdm/mcp-server-gemini-bridge) 🐍 ☁️ - Bridge to Google Gemini API. Access Gemini Pro and Flash models through MCP.
- [jaspertvdm/mcp-server-ollama-bridge](https://github.com/jaspertvdm/mcp-server-ollama-bridge) 🐍 🏠 - Bridge to local Ollama LLM server. Run Llama, Mistral, Qwen and other local models through MCP.
- [jaspertvdm/mcp-server-openai-bridge](https://github.com/jaspertvdm/mcp-server-openai-bridge) 🐍 ☁️ - Bridge to OpenAI API. Access GPT-4, GPT-4o and other OpenAI models through MCP.
- [julien040/anyquery](https://github.com/julien040/anyquery) 🏎️ 🏠 ☁️ - Query more than 40 apps with one binary using SQL. It can also connect to your PostgreSQL, MySQL, or SQLite compatible database. Local-first and private by design.

```markdown
- [NotAnEntry](https://example.com/fenced) - inside a fence
```

