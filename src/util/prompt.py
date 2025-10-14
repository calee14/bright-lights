ANALYZE_PROMPT = """Analyze the provided SPY intraday candlestick chart (1-hour and/or 3-hour timeframe) for the following popular candlestick patterns: Hammer, Inverse Hammer, Bullish Engulfing, Piercing Line, Morning Star, Three White Soldiers, Hanging Man, Shooting Star, Bearish Engulfing, Dark Cloud Cover, Evening Star, Three Black Crows, Abandoned Baby, Harami (Bullish/Bearish), Kicking, Marubozu, Rising Three Methods, Falling Three Methods, Doji (including Dragonfly, Gravestone, Long-Legged), Spinning Top, Tasuki Gap, and Three Inside Up/Down. Additionally, identify any Fair Value Gaps (FVGs) by detecting three-candle formations where the third candle’s low (for bullish FVG) or high (for bearish FVG) does not overlap the first candle’s high or low, respectively. Use FVGs as confluence to strengthen or filter candlestick pattern signals (e.g., a Bullish Engulfing at a bullish FVG is a stronger buy signal). Include volume trends in your analysis, noting whether volume is increasing, decreasing, or stable to confirm pattern strength. Identify key support and resistance levels based on recent price action (e.g., swing highs/lows or zones where price consolidates). Predict whether the price will move up or down in the next 1 hour, providing a probability percentage (e.g., 70% chance of upside) and a concise rationale (at most 300 words) explaining how the candlestick patterns, FVGs, volume trends, and support/resistance levels inform your prediction. If analyzing a 1-hour chart, compare it to a 3-hour chart (if provided) to adjust your prediction based on broader context. Output the response in a structured format with sections for Patterns Detected, FVG Analysis, Volume Trends, Support/Resistance, and Price Prediction."""


def generate_summary_prompt(responses):
    prompt = f"""Analyze these multiple LLM responses on SPY candlestick chart patterns, FVGs, volume trends, and price predictions. Provide a concise, intelligent synthesis that extracts:

1. **Dominant technical signals** - The most frequently identified patterns and FVGs across responses, with their directional bias
2. **Critical price levels** - Support/resistance zones showing strongest consensus
3. **Volume-price alignment** - Key volume trends reinforcing or contradicting price signals
4. **Consensus forecast** - Predominant price direction, timeframe, and confidence drivers
5. **Highest-conviction setup** - The single most consistent analysis across responses

Focus on convergence across responses. Deliver sharp, digestible insights prioritizing signal strength over exhaustive detail. Structure as 5 bullet points, each 2-3 sentences maximum.

**Responses to analyze (seperated by ---):**
{"\n---\n".join(responses)}"""

    return prompt
