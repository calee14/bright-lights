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

STAGE3_PREDICTION = """Based on your Stage 1 pattern detection and Stage 2 confluence analysis, make your prediction.

**SYNTHESIS PROCESS:**

1. **Prioritize**: Which pattern(s) have the highest confluence scores (4-5)?

2. **Conflicts**: Are there contradicting signals? If yes:
   - Which timeframe takes priority?
   - Which confluence factors are stronger?

3. **Direction Bias**: Based on the strongest signals, what's the primary directional bias?

**PREDICTION FORMAT:**

## Primary Setup
[Describe the highest-confluence pattern and its supporting factors]

## Direction
**Up** / **Down** / **Neutral**

## Confidence
[X]% (Justify based on confluence score and agreement of factors)

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
- Which pattern has the best setup?
- What confluence factors support it most strongly?
- Are there any contradicting factors you're overriding? Why?

**CRITICAL RULES:**
- If no pattern scores above 3, state "No high-conviction setup"
- If contradicting 4+ scored patterns exist, state "Mixed signals, no clear bias"
- Be honest about uncertainty - don't force a prediction
- Focus on what's CLEARLY visible in the charts
- ALWAYS provide specific numeric price levels for upper and lower bounds
"""


def generate_final_alert_prompt(
    consensus: str, agreement_pct: float, avg_confidence: float, analyst_summaries: str
) -> str:
    """Generate prompt for final summary aggregation"""

    return f"""You are aggregating predictions from 3 independent technical analysts who each completed a thorough 3-stage analysis of QQQ charts.

**CONSENSUS DATA:**
- Direction: {consensus}
- Analyst Agreement: {agreement_pct:.0f}%
- Average Confidence: {avg_confidence:.0f}%

**INDIVIDUAL ANALYST PREDICTIONS:**
{analyst_summaries}

**YOUR TASK:**
Create a concise final summary (max 250 words) that:

1. **States the consensus** clearly upfront
2. **Highlights the strongest confluence factors** mentioned by multiple analysts
3. **Notes any disagreements** and why certain analysts differed
4. **Provides actionable insight** - what should a trader focus on?
5. **Consolidates price targets** - average or use the most commonly cited upper/lower bounds

**OUTPUT FORMAT:**

## Market Outlook: {consensus}
**Confidence Level**: {avg_confidence:.0f}% | **Analyst Agreement**: {agreement_pct:.0f}%

### Price Targets
**Upper Bound**: [Consolidated upper price target]
**Lower Bound**: [Consolidated lower price target]

### Key Factors
[2-3 bullet points of the most important confluence factors]

### Trade Consideration
[1-2 sentences on invalidation level and timeframe]

### Analyst Notes
[Brief mention of any important disagreements or caveats]

**CRITICAL**: 
- Be concise and actionable
- Don't repeat everything - synthesize the MOST important points
- If agreement is low (<67%), emphasize caution
- Focus on factors mentioned by multiple analysts
- MUST include specific numeric price levels for Upper Bound and Lower Bound
- If analysts provided different price targets, average them or use the most conservative range
"""

