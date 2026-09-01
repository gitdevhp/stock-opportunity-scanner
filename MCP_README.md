# Optional MCP layer

The scheduled Discord scanner does not require MCP.

If you later want an MCP-capable assistant to query the scanner, expose read-only tools such as:

- `scan_market()`
- `analyze_stock(ticker)`
- `get_last_report()`
- `get_watchlist()`
- `set_capital(amount)`
- `explain_score(ticker)`

Recommended architecture:

Assistant
  -> MCP server
  -> scanner.py / cached reports
  -> yfinance data

Keep trading execution OUT of this initial MCP layer. Make it read-only until the scanner has been validated extensively.

A future MCP server can be written with the official Python MCP SDK, but adding it now is unnecessary complexity for the free scheduled Discord system.
