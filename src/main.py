from typing import List
from anthropic.types import ContentBlock
from src.util.feed import get_yahoo_finance_data, parse_yahoo_data
from src.util.plot import plot_recent_candlesticks
from src.util.vibe import chat, build_msg
from src.util.prompt import ANALYZE_PROMPT, generate_summary_prompt
from concurrent.futures import ThreadPoolExecutor


def pack_member(prompt: str, charts: list[str] = []) -> List[ContentBlock]:
    msg = build_msg([prompt], charts)

    res = chat(msg)

    return res


def run_pack(prompt: str, charts: list[str], num_calls: int = 7):
    with ThreadPoolExecutor(max_workers=7) as executor:
        futures = [
            executor.submit(pack_member, prompt, charts) for _ in range(num_calls)
        ]

        results = [future.result() for future in futures]

    return results


def ticker_scent(ticker: str):
    charts = ["charts/tminus60.jpeg", "charts/tminus360.jpeg"]
    tminus60 = parse_yahoo_data(
        get_yahoo_finance_data(ticker, lookback=3600, interval="1m")
    )

    plot_recent_candlesticks(tminus60, last_n_periods=len(tminus60), filename=charts[0])

    tminus360 = parse_yahoo_data(
        get_yahoo_finance_data(ticker, lookback=10800, interval="5m")
    )

    plot_recent_candlesticks(
        tminus360, last_n_periods=len(tminus360), filename=charts[1]
    )

    return charts


if __name__ == "__main__":
    files = ticker_scent("SPY")

    results = run_pack(ANALYZE_PROMPT, files, num_calls=7)

    analysis = []
    for r in results:
        cb = r[0]
        analysis.append(cb.text)

    summary_prompt = generate_summary_prompt(analysis)

    final_summary = pack_member(summary_prompt)

    print(final_summary)
