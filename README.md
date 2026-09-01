# Technical Stock Opportunity Scanner

A free, rule-based stock scanner that sends morning, hourly intraday, and evening technical opportunity reports to Discord.

## What it does

The scanner ranks stocks using the framework we defined:

### Major signals — 80%
- 50/200-day moving-average relationship: 20%
- Repeated horizontal support/resistance: 20%
- Trend/channel or converging-channel structure: 20%
- Breakout/retest + volume confirmation: 20%

### Secondary signals — 20%
- Bollinger Bands: 5%
- RSI: 5%
- Fibonacci retracement: 5%
- MACD + volatility: 5%

It also applies penalties for:
- very weak liquidity
- extreme short-term extension
- imminent earnings when detectable
- broken major support
- excessive volatility

For each high-scoring setup it calculates:
- preferred limit entry
- breakout confirmation level
- invalidation/stop level
- profit target
- fractional share quantity using the configured capital
- Observe / Reason / Response summary

It sends the report to a Discord webhook.

## Cost

The scanner itself uses free/open-source Python packages and yfinance. yfinance is intended for personal/research use and is not affiliated with Yahoo Finance; review its terms before using the data commercially.

For GitHub Actions, a public repository is the simplest zero-cost route. GitHub documents scheduled workflows and hosted runners here:
https://docs.github.com/en/actions

Discord webhooks are built into Discord and can receive HTTP POST messages.

## Important

This is a research/alerting system. It does NOT place trades and it does NOT guarantee profits.

## Quick start

### 1. Create a Discord webhook

In your Discord server:
Server Settings -> Integrations -> Webhooks -> New Webhook

Copy the webhook URL.

Do NOT put the URL directly into source code.

### 2. Create a GitHub repository

Recommended name:

`stock-opportunity-scanner`

A public repo is easiest for keeping the GitHub Actions cost at $0. Do not commit secrets.

### 3. Upload these files

Copy the project contents into the repository.

### 4. Add the Discord secret

GitHub:
Repository -> Settings -> Secrets and variables -> Actions -> New repository secret

Name:
`DISCORD_WEBHOOK_URL`

Value:
your Discord webhook URL

### 5. Adjust capital

Edit `config.json`:

```json
{
  "capital": 333.0
}
```

This is only used to calculate hypothetical fractional-share sizing.

### 6. Run manually

GitHub:
Actions -> Stock Opportunity Scanner -> Run workflow

The workflow also runs twice daily.

### 7. Intraday schedule

On trading days, GitHub Actions launches hourly candidate runs and `scanner.py` sends only during these America/Chicago local times:

- 09:00
- 10:00
- 11:00
- 12:00
- 13:00
- 14:00
- 15:00

The workflow includes both UTC schedules needed for Central Time daylight-saving changes. GitHub scheduled workflows can occasionally start late, so the scanner accepts a short timing window. This is suitable for swing-trade opportunity monitoring, not latency-sensitive execution.

The current scanner is intentionally conservative about alerts: it reports qualifying setups but does not place trades.

### 8. Local test

Install Python 3.11+:

```bash
pip install -r requirements.txt
python scanner.py --mode evening
```

For a local Discord test:

```bash
export DISCORD_WEBHOOK_URL="YOUR_WEBHOOK"
python scanner.py --mode evening --send
```

Windows PowerShell:

```powershell
$env:DISCORD_WEBHOOK_URL="YOUR_WEBHOOK"
python scanner.py --mode evening --send
```

## Schedule

GitHub Actions cron is UTC. The workflow intentionally runs four candidate times and the Python script checks America/Chicago local time so daylight-saving changes do not require editing the workflow.

Default target windows:
- Morning: ~8:35 AM Central
- Evening: ~5:05 PM Central

The workflow may start at nearby UTC times, but the script exits without sending if the local time is outside the target window.

## Strategy

The scanner does NOT force a buy.

Signal states:
- 85-100: HIGH CONVICTION
- 75-84: WATCH / PREPARE LIMIT
- 65-74: DEVELOPING
- below 65: IGNORE

A stock can score highly but still receive `WAIT FOR CONFIRMATION` if its entry would require chasing price.

## Data universe

By default, the scanner pulls S&P 500 and Nasdaq-100 constituents from public Wikipedia tables, then adds a small list of important non-index names from `config.json`.

You can replace the universe with your own `tickers.txt` file.

## MCP

MCP is intentionally separated from the scheduled scanner.

The free scheduled system does not need MCP. Later, an MCP server can expose:
- scan()
- analyze_stock(ticker)
- get_watchlist()
- get_last_report()
- set_capital()
- explain_score(ticker)

This lets an MCP-capable assistant query the same scanner without changing the automated Discord process.

