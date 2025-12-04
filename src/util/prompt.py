# src/util/prompt.py

STAGE1_TREND_ANALYSIS = """You are analyzing QQQ candlestick chart images. Your ONLY task in this stage is to characterize the current trend.

**STEP 1: CHART ORIENTATION**
- What timeframe is shown? (check x-axis labels)
- How many candles are visible?
- What's the price range (high to low)?

**STEP 2: TREND IDENTIFICATION**
Examine the last 20-30 candles (rightmost section):

**A. Trend Direction**
- Is price making higher highs AND higher lows? → Uptrend
- Is price making lower highs AND lower lows? → Downtrend  
- Is price moving sideways in a range? → Sideways/Ranging

**B. Trend Strength (0-100 scale)**
Count the last 20 candles and score:

*Directional Consistency (0-40 points):*
- How many candles close in the trend direction vs against it?
- 18-20 directional candles = 40 pts | 15-17 = 30 pts | 12-14 = 20 pts | <12 = 10 pts

*Sequential Structure (0-30 points):*
- Uptrend: Count consecutive higher highs + higher lows
- Downtrend: Count consecutive lower highs + lower lows
- 8+ clean sequences = 30 pts | 6-7 = 24 pts | 4-5 = 18 pts | 2-3 = 12 pts

*Momentum/Angle (0-30 points):*
- Steep, aggressive moves = 30 pts
- Moderate, steady moves = 20 pts
- Shallow, gradual moves = 10 pts
- Flat or erratic = 5 pts

**Total Trend Strength = A + B + C**

**C. Trend Quality**
- Clean: Few pullbacks, smooth progression
- Choppy: Many back-and-forth moves, indecision
- Pausing: Consolidating after a move

**D. Trend Duration**
- How many candles has this trend been in place?
- Is it young (just starting) or mature (extended)?

**DO NOT:**
- Predict where price will go
- Identify support/resistance levels yet
- Suggest trades or positions

**OUTPUT FORMAT:**
## Chart Context
Timeframe: [1m, 5m, etc.]
Candles Visible: [count]
Price Range: $[low] - $[high]

## Trend Direction
[Uptrend / Downtrend / Sideways]

## Trend Strength
**Total Score**: [X]/100 points

**Breakdown:**
- Directional Consistency: [X]/40 points ([Y] of 20 candles moving with trend)
- Sequential Structure: [X]/30 points ([Y] clean higher-high/low sequences)
- Momentum/Angle: [X]/30 points (Steep/Moderate/Shallow)

## Trend Quality
[Clean / Choppy / Pausing]
[Describe the character of price movement]

## Trend Duration
[X candles in current trend]
[Young/Developing/Mature]

## Visual Observations
[Any notable characteristics: steady climb, volatile swings, consolidation phases, etc.]
"""

STAGE2_LEVELS_ANALYSIS = """You have characterized the trend. Now identify key support and resistance levels and assess their strength.

**Review the trend analysis from Stage 1, then assess:**

**A. SUPPORT LEVELS**
Identify 2-4 support levels where price has bounced or found buyers:
- Recent swing lows
- Prior consolidation zones
- Areas where price repeatedly held

For each support level:
1. **Price**: Exact or approximate level
2. **Strength Rating (0-100)**:
   - Number of touches: 5+ touches = 30 pts | 3-4 = 20 pts | 1-2 = 10 pts
   - Recency: Tested recently = 30 pts | Within 50 candles = 20 pts | Older = 10 pts
   - Volume profile: High volume area = 40 pts | Medium = 25 pts | Low = 10 pts
3. **Distance from Current Price**: [X] points or [Y]%
4. **Hold Probability**: Based on strength rating and current trend
   - 90-100 strength = 85-95% hold probability
   - 70-89 strength = 65-84% hold probability
   - 50-69 strength = 45-64% hold probability
   - Below 50 strength = 25-44% hold probability

**B. RESISTANCE LEVELS**
Identify 2-4 resistance levels where price has stalled or found sellers:
- Recent swing highs
- Prior consolidation zones
- Areas where price repeatedly rejected

For each resistance level:
1. **Price**: Exact or approximate level
2. **Strength Rating (0-100)**: Same scoring as support
3. **Distance from Current Price**: [X] points or [Y]%
4. **Hold Probability**: Based on strength rating and current trend

**C. LEVEL INTERACTION**
- Is price currently near any key level?
- Which level is most critical to watch right now?
- In the context of the trend (Stage 1), which levels are most relevant?

**D. VOLUME ANALYSIS AT LEVELS**
- Compare volume when price touches levels vs average volume
- Are there volume spikes at any key levels?
- Does volume support the strength ratings you assigned?

**DO NOT:**
- Predict if levels will break or hold (only provide probabilities)
- Suggest trades based on levels
- Make directional calls

**OUTPUT FORMAT:**
## Support Levels (Ordered by proximity to current price)

### Support 1: $[price]
- Strength Rating: [X]/100
  - Touches: [count] ([points])
  - Recency: [description] ([points])
  - Volume: [High/Medium/Low] ([points])
- Distance: $[X] below / [Y]% below current
- Hold Probability: [Z]%
- Notes: [Why this level matters]

[Repeat for each support level]

## Resistance Levels (Ordered by proximity to current price)

### Resistance 1: $[price]
- Strength Rating: [X]/100
  - Touches: [count] ([points])
  - Recency: [description] ([points])
  - Volume: [High/Medium/Low] ([points])
- Distance: $[X] above / [Y]% above current
- Hold Probability: [Z]%
- Notes: [Why this level matters]

[Repeat for each resistance level]

## Current Level Interaction
[Describe which level(s) are being tested or approached right now]

## Most Critical Level
[Which single level matters most given the current trend and price position]

## Volume Observations
[How volume confirms or contradicts the level strength ratings]
"""

STAGE3_REVERSAL_WATCH = """Based on your trend analysis (Stage 1) and key levels (Stage 2), assess the potential for trend reversal.

**SYNTHESIS PROCESS:**

**A. REVERSAL INDICATORS**
Look for signs that the current trend may be weakening or reversing:

1. **Momentum Divergence** (0-25 points)
   - Price making new highs/lows but with smaller candles = 25 pts
   - Slowing momentum visible = 15 pts
   - Momentum still strong = 0 pts

2. **Failed Level Tests** (0-25 points)
   - Multiple rejections at resistance (uptrend) or support (downtrend) = 25 pts
   - One clear rejection = 15 pts
   - No clear rejection yet = 0 pts

3. **Reversal Candlestick Patterns** (0-25 points)
   - Strong reversal patterns present (engulfing, shooting star, hammer at level) = 25 pts
   - Weak reversal patterns present = 15 pts
   - No reversal patterns = 0 pts

4. **Trend Structure Breaking** (0-25 points)
   - Uptrend: Lower high formed = 25 pts | Lower low formed = 25 pts
   - Downtrend: Higher low formed = 25 pts | Higher high formed = 25 pts
   - Trend structure intact = 0 pts

**Total Reversal Score = Sum of 1-4 (Max 100)**

**B. REVERSAL PROBABILITY**
Based on total reversal score:
- 75-100 points = High probability (60-80%) - Multiple strong signals
- 50-74 points = Moderate probability (35-59%) - Some signals present
- 25-49 points = Low probability (15-34%) - Early/weak signals
- 0-24 points = Very low probability (0-14%) - No meaningful signals

**C. INVALIDATION CRITERIA**
What would prove that NO reversal is happening?
- For potential uptrend reversal: "Strong break above $[X] resistance"
- For potential downtrend reversal: "Strong break below $[X] support"

**D. CONFIRMATION CRITERIA**
What would CONFIRM a reversal is actually happening?
- Specific price level breaks
- Specific candlestick patterns completing
- Trend structure definitively changing

**DO NOT:**
- Tell the user to trade based on reversal signals
- Predict with certainty whether reversal will happen
- Provide entry/exit points

**OUTPUT FORMAT:**
## Reversal Assessment

### Current Trend Status
[Restate from Stage 1: Direction and Strength]

### Reversal Score: [X]/100 points

**Breakdown:**
- Momentum Divergence: [X]/25 pts - [Description]
- Failed Level Tests: [X]/25 pts - [Description]
- Reversal Patterns: [X]/25 pts - [Description]
- Trend Structure: [X]/25 pts - [Description]

### Reversal Probability: [X]%
[High/Moderate/Low/Very Low] - [Brief explanation]

### Warning Signs to Monitor
[List 2-3 specific things to watch that would signal reversal is developing]

### Invalidation Level
"Reversal scenario invalidated if price [moves above/below] $[X]"
[Explain why this level]

### Confirmation Criteria
"Reversal confirmed if:"
- [Criterion 1]
- [Criterion 2]
- [Criterion 3]

### Context from Key Levels
[How the Stage 2 support/resistance levels factor into reversal potential]

### Summary
[2-3 sentences: Is reversal likely? What's the evidence? What should be monitored?]

**CRITICAL**: This is information only. Present the evidence objectively without suggesting trading actions.
"""


def generate_final_summary_prompt(
    trend_consensus: dict,
    levels_consensus: dict,
    reversal_consensus: dict,
    analyst_summaries: str,
) -> str:
    """Generate prompt for final informational summary"""

    return f"""You are aggregating market analysis from 2 independent technical analysts who each completed a thorough 3-stage analysis of QQQ charts.

**CONSENSUS DATA:**

**Trend Analysis:**
- Direction Agreement: {trend_consensus["direction_agreement"]:.0f}%
- Average Trend Strength: {trend_consensus["avg_strength"]:.0f}/100
- Quality: {trend_consensus["quality"]}

**Key Levels:**
- Critical Support: ${levels_consensus["strongest_support"]:.2f} (Avg {levels_consensus["support_strength"]:.0f}/100 strength, {levels_consensus["support_hold_prob"]:.0f}% hold probability)
- Critical Resistance: ${levels_consensus["strongest_resistance"]:.2f} (Avg {levels_consensus["resistance_strength"]:.0f}/100 strength, {levels_consensus["resistance_hold_prob"]:.0f}% hold probability)
- Current Price: ${levels_consensus["current_price"]:.2f}

**Reversal Watch:**
- Average Reversal Score: {reversal_consensus["avg_score"]:.0f}/100
- Reversal Probability: {reversal_consensus["avg_probability"]:.0f}%

**INDIVIDUAL ANALYST ANALYSES:**
{analyst_summaries}

**YOUR TASK:**
Create a concise informational summary (max 300 words) that helps a trader understand the current market state WITHOUT telling them what to do.

Focus on:
1. **What IS happening** (trend state, key levels being tested)
2. **What COULD happen** (probabilities, scenarios to monitor)
3. **What would CHANGE the picture** (invalidation and confirmation levels)

**OUTPUT FORMAT:**

## Market State Summary

### Trend Status
**Direction**: [Consensus direction] (Analyst agreement: [X]%)
**Strength**: [X]/100 - [Strong/Moderate/Weak/No Clear Trend]
**Character**: [Clean/Choppy/Pausing]
**Duration**: [How long this trend has been in place]

### Key Levels to Monitor

**Critical Resistance**: $[X]
- Strength: [Y]/100
- Hold Probability: [Z]%
- Distance: $[A] above current ([B]%)

**Current Price**: $[X]

**Critical Support**: $[X]
- Strength: [Y]/100
- Hold Probability: [Z]%
- Distance: $[A] below current ([B]%)

### Reversal Watch

**Reversal Score**: [X]/100
**Probability**: [Y]% ([High/Moderate/Low/Very Low])

**Warning Signs Present**:
[List key reversal indicators if any]

**What Would Invalidate Reversal Concerns**:
[Specific price level or action]

**What Would Confirm Reversal**:
[Specific criteria]

### What to Monitor

1. [Most important thing to watch right now]
2. [Second most important]
3. [Third most important]

### Analyst Agreement Notes
[Brief mention of where analysts agreed strongly and any significant disagreements]

**CRITICAL RULES:**
- This is INFORMATION, not trading advice
- Use "if price does X, then Y becomes more/less likely" language
- Present probabilities and scenarios objectively
- No "you should" or "trade this" language
- Focus on facts and likelihoods, not recommendations
"""
