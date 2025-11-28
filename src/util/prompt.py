# src/util/prompt.py

STAGE1_PATTERN_RECOGNITION = """You are analyzing QQQ candlestick chart images. Your ONLY task in this stage is to identify patterns.

**STEP 1: CHART ORIENTATION**
- What timeframe is shown? (check x-axis labels)
- How many candles are visible?
- What's the overall trend in the visible window?

**STEP 2: PATTERN DETECTION**
Focus on the most recent 15-20 candles (rightmost). Identify these patterns:

**High Priority (Easy to spot):**
- Engulfing (Bullish/Bearish)
- Hammer / Shooting Star
- Doji (Dragonfly, Gravestone, Long-Legged)
- Marubozu
- Three White Soldiers / Three Black Crows

**Medium Priority (If clear):**
- Morning Star / Evening Star
- Piercing Line / Dark Cloud Cover
- Harami (Bullish/Bearish)

For each pattern found, describe:
1. **Location**: "3rd candle from right edge" or "near the middle section after decline"
2. **Quality**: Strong/Moderate/Weak (based on body size, wick proportions, clarity)
3. **Certainty**: High/Medium/Low (if image resolution makes it ambiguous, say so)

**DO NOT:**
- Make predictions yet
- Analyze volume
- Identify support/resistance
- Calculate probabilities

**OUTPUT FORMAT:**
## Chart Context
[Timeframe, candle count, overall trend]

## Patterns Detected
[List each pattern with location, quality, and certainty]

## Ambiguities
[Any patterns you're unsure about due to image quality]
"""

STAGE2_CONFLUENCE_ANALYSIS = """You have already identified candlestick patterns. Now analyze the context and confluence factors.

**Review the patterns from Stage 1, then assess:**

**A. FAIR VALUE GAPS (FVGs)**
Look for 3-candle sequences where there's a visible gap:
- Bullish FVG: Gap between candle 1's high and candle 3's low
- Bearish FVG: Gap between candle 1's low and candle 3's high

For each FVG found:
- Describe location
- Is it filled or unfilled?
- Does it align with any patterns from Stage 1?

**B. VOLUME ANALYSIS**
Examine the volume bars (bottom of chart):
- Compare volume on pattern candles vs. the recent average
- Is volume expanding, contracting, or neutral?
- Does volume confirm or contradict the pattern's typical interpretation?

For each Stage 1 pattern, state: "Volume confirms" / "Volume neutral" / "Volume contradicts"

**C. SUPPORT & RESISTANCE**
Visually identify 2-3 key horizontal levels where:
- Price repeatedly bounced (support)
- Price repeatedly stalled (resistance)
- Price consolidated

Note which Stage 1 patterns are forming near these levels.

**D. MULTI-TIMEFRAME ALIGNMENT** (if both 1H and 3H/5H charts provided)
- Does the shorter timeframe pattern align with the longer timeframe trend?
- Are patterns forming at significant longer-timeframe levels?

**CONFLUENCE SCORING**
For each Stage 1 pattern, assign a score (1-5):
- 5 = Pattern + FVG + Volume + S/R all aligned
- 4 = Pattern + 2 confirming factors
- 3 = Pattern + 1 confirming factor
- 2 = Pattern only, no confirmation
- 1 = Pattern with contradicting factors

**OUTPUT FORMAT:**
## FVG Analysis
[List all FVGs with details]

## Volume Assessment
[Overall volume trend + per-pattern notes]

## Support/Resistance Levels
[Describe 2-3 key levels]

## Multi-Timeframe Check
[Alignment or divergence notes]

## Confluence Scores
[Each Stage 1 pattern with score 1-5 and reasoning]
"""

STAGE3_PREDICTION = """Based on your Stage 1 pattern detection and Stage 2 confluence analysis, make your prediction with emphasis on TREND and BREAKOUT analysis.

**SYNTHESIS PROCESS:**

1. **Trend Analysis** - Use this QUANTITATIVE FRAMEWORK:
   
   Examine the last 15-20 candles and score based on these factors:
   
   **A. Directional Consistency (0-40 points)**
   - Count candles moving in primary direction vs. total candles
   - 20/20 candles = 40 points | 18/20 = 36 points | 15/20 = 30 points | 12/20 = 24 points | <10/20 = <20 points
   
   **B. Higher Highs/Lower Lows Sequence (0-30 points)**
   - Uptrend: Count consecutive higher highs AND higher lows
   - Downtrend: Count consecutive lower highs AND lower lows
   - 8+ clean sequences = 30 points | 6-7 sequences = 24 points | 4-5 sequences = 18 points | 2-3 sequences = 12 points
   
   **C. Angle/Momentum (0-30 points)**
   - Steep, aggressive movement = 30 points
   - Moderate, steady movement = 20 points
   - Shallow, gradual movement = 10 points
   - Flat or erratic = 0 points
   
   **TOTAL TREND STRENGTH = Sum of A + B + C**
   - 90-100 points = 90-100% strength (Very strong trend)
   - 70-89 points = 70-89% strength (Strong trend)
   - 50-69 points = 50-69% strength (Moderate trend)
   - 30-49 points = 30-49% strength (Weak trend)
   - 0-29 points = 0-29% strength (No clear trend/sideways)

2. **Breakout Analysis** - Use this QUANTITATIVE FRAMEWORK:
   
   **A. Level Break Confirmation (0-35 points)**
   - Clean break of major S/R with body close beyond level = 35 points
   - Break with close near level (testing) = 25 points
   - Wick through level only = 15 points
   - No clear level break = 0 points
   
   **B. Volume Confirmation (0-30 points)**
   - Volume 2x+ higher than recent average = 30 points
   - Volume 1.5-2x average = 22 points
   - Volume 1.2-1.5x average = 15 points
   - Volume at or below average = 0 points
   
   **C. Follow-Through Strength (0-35 points)**
   - 3+ strong candles continuing in breakout direction = 35 points
   - 2 strong candles with momentum = 25 points
   - 1 candle then consolidation = 15 points
   - Immediate reversal/failed breakout = 0 points
   
   **TOTAL BREAKOUT STRENGTH = Sum of A + B + C**
   - 85-100 points = 85-100% strength (Explosive breakout)
   - 65-84 points = 65-84% strength (Strong breakout)
   - 45-64 points = 45-64% strength (Moderate breakout)
   - 25-44 points = 25-44% strength (Weak breakout)
   - 0-24 points = 0-24% strength (No breakout/false breakout)

3. **Pattern Confluence**: Which pattern(s) have the highest confluence scores (4-5)?

4. **Direction Bias**: Based on trend + breakout + patterns, what's the primary directional bias?

**PREDICTION FORMAT:**

## Trend Assessment
**Trend Exists**: Yes / No
**Trend Direction**: Up / Down / Sideways
**Trend Strength**: [X]%

**Scoring Breakdown:**
- Directional Consistency: [X]/40 points ([Y]/20 candles in primary direction)
- Higher Highs/Lower Lows: [X]/30 points ([Y] consecutive sequences)
- Angle/Momentum: [X]/30 points (Steep/Moderate/Shallow/Flat)
- **Total: [X]/100 points = [X]% Trend Strength**

## Breakout Assessment
**Breakout Exists**: Yes / No
**Breakout Direction**: Up / Down / None
**Breakout Strength**: [X]%

**Scoring Breakdown:**
- Level Break Confirmation: [X]/35 points (Clean/Testing/Wick-only/None)
- Volume Confirmation: [X]/30 points ([Y]x average volume)
- Follow-Through Strength: [X]/35 points ([Y] continuation candles)
- **Total: [X]/100 points = [X]% Breakout Strength**

## Primary Setup
[Describe the highest-confluence pattern and its supporting factors, with emphasis on how it aligns with trend/breakout]

## Direction
**Up** / **Down** / **Neutral**

## Confidence
[X]% (Justify based on trend strength + breakout strength + confluence score)

## Price Targets
**Upper Bound**: [Specific price level - resistance or pattern target for upside]
**Lower Bound**: [Specific price level - support or invalidation for downside]

(Note: Provide actual numeric price levels based on visible chart levels. For UP direction, upper bound is the target; for DOWN direction, lower bound is the target. The other bound serves as invalidation/stop level.)

## Key Level
**Invalidation**: If price closes below/above [describe specific level or pattern], this setup is invalid

## Timeframe
Next 1 hour

## Rationale (Max 200 words)
Explain the 2-3 most important factors:
- What is the trend telling us?
- Is there a confirmed breakout?
- Which pattern has the best setup?
- What confluence factors support it most strongly?
- Are there any contradicting factors you're overriding? Why?

**CRITICAL RULES:**
- Use the QUANTITATIVE SCORING FRAMEWORK above - be specific with points
- Trend Strength: Add up A+B+C points, convert total to percentage (e.g., 75 points = 75%)
- Breakout Strength: Add up A+B+C points, convert total to percentage (e.g., 60 points = 60%)
- If trend strength < 30% AND no pattern scores above 3, state "No high-conviction setup"
- Focus on TREND and BREAKOUT as primary signals, patterns as confirmation
- Be honest about uncertainty - don't force a prediction
- Focus on what's CLEARLY visible in the charts
- ALWAYS provide specific numeric price levels for upper and lower bounds
- Show your scoring work in the "Scoring Breakdown" sections
"""


def generate_final_alert_prompt(
    consensus: str,
    agreement_pct: float,
    avg_confidence: float,
    analyst_summaries: str,
    avg_trend_strength: float = 0,
    avg_breakout_strength: float = 0,
) -> str:
    """Generate prompt for final summary aggregation"""

    return f"""You are aggregating predictions from 2 independent technical analysts who each completed a thorough 3-stage analysis of QQQ charts.

**CONSENSUS DATA:**
- Direction: {consensus}
- Analyst Agreement: {agreement_pct:.0f}%
- Average Confidence: {avg_confidence:.0f}%
- Average Trend Strength: {avg_trend_strength:.0f}%
- Average Breakout Strength: {avg_breakout_strength:.0f}%

**INDIVIDUAL ANALYST PREDICTIONS:**
{analyst_summaries}

**YOUR TASK:**
Create a concise final summary (max 250 words) that:

1. **States the consensus** clearly upfront with trend/breakout context
2. **Highlights trend and breakout signals** - are they aligned?
3. **Notes the strongest confluence factors** mentioned by multiple analysts
4. **Notes any disagreements** and why certain analysts differed
5. **Provides actionable insight** - what should a trader focus on?
6. **Consolidates price targets** - average or use the most commonly cited upper/lower bounds

**OUTPUT FORMAT:**

## Market Outlook: {consensus}
**Confidence Level**: {avg_confidence:.0f}% | **Analyst Agreement**: {agreement_pct:.0f}%
**Trend Strength**: {avg_trend_strength:.0f}% | **Breakout Strength**: {avg_breakout_strength:.0f}%

### Price Targets
**Upper Bound**: [Consolidated upper price target]
**Lower Bound**: [Consolidated lower price target]

### Key Factors
[2-3 bullet points focusing on trend/breakout signals and most important confluence factors]

### Trade Consideration
[1-2 sentences on invalidation level and timeframe]

### Analyst Notes
[Brief mention of any important disagreements or caveats]

**CRITICAL**: 
- Be concise and actionable
- Emphasize trend and breakout strength in your assessment
- Don't repeat everything - synthesize the MOST important points
- If agreement is low (<67%), emphasize caution
- Focus on factors mentioned by multiple analysts
- MUST include specific numeric price levels for Upper Bound and Lower Bound
- If analysts provided different price targets, average them or use the most conservative range
"""
