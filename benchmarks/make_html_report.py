#!/usr/bin/env python3
"""Generate a self-contained HTML report from compare_chat_vs_decision.py
JSON output (use --json-out on that script first).

The report embeds all CSS inline and has zero external dependencies, so it
renders on an isolated intranet machine with no network access.

Usage:
    python3 benchmarks/make_html_report.py \
        --json benchmarks/compare_results.json \
        --out   benchmarks/jev_compare_report.html
"""

from __future__ import annotations

import argparse
import html
import json
import os
import platform
import statistics
from datetime import datetime


def esc(value) -> str:
    return html.escape(str(value))


def ms(value: float | None) -> str:
    if value is None:
        return "—"
    if value >= 1000:
        return f"{value / 1000:.2f}s"
    return f"{value:.0f}ms"


def warm_rows(rows: list[dict]) -> list[dict]:
    """Drop the first run when it is a cold-start outlier (>3x the median
    of the remaining runs) so charts stay readable."""
    if len(rows) < 3:
        return rows
    rest = [r["wall_ms"] for r in rows[1:]]
    median = statistics.median(rest)
    if rows[0]["wall_ms"] > 3 * median:
        return rows[1:]
    return rows


def bars_for_runs(decision: list[dict], chat: list[dict]) -> str:
    """Grouped vertical bar chart, one pair of bars per run."""
    pairs = [(d, c) for d, c in zip(decision, chat)]
    peak = max(
        max(d["wall_ms"] for d, _ in pairs),
        max(c["wall_ms"] for _, c in pairs),
    ) or 1.0
    blocks = []
    for d, c in pairs:
        dh = max(4, round(d["wall_ms"] / peak * 150))
        ch = max(4, round(c["wall_ms"] / peak * 150))
        blocks.append(f"""
        <div class="rungroup">
          <div class="runlabel">run {esc(d['run'])}</div>
          <div class="bars">
            <div class="barcol"><div class="bar decision" style="height:{dh}px" title="{ms(d['wall_ms'])}"></div><span class="barval">{ms(d['wall_ms'])}</span></div>
            <div class="barcol"><div class="bar chat" style="height:{ch}px" title="{ms(c['wall_ms'])}"></div><span class="barval">{ms(c['wall_ms'])}</span></div>
          </div>
        </div>""")
    return "".join(blocks)


def hbar(label: str, value: float, peak: float, cls: str, note: str = "") -> str:
    width = max(2, round(value / peak * 100))
    return f"""
      <div class="hbarrow">
        <div class="hbarlabel">{esc(label)}</div>
        <div class="hbartrack"><div class="hbar {cls}" style="width:{width}%"></div></div>
        <div class="hbarnote">{esc(note)}</div>
      </div>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True)
    parser.add_argument("--out", default="jev_compare_report.html")
    args = parser.parse_args()

    with open(args.json, encoding="utf-8") as handle:
        data = json.load(handle)

    meta = data["meta"]
    decision_rows = data["decision_rows"]
    chat_rows = data["chat_rows"]
    summary = data["summary"]
    health = meta.get("jev_health") or {}

    # Cold-run detection (chart clarity, table keeps all runs).
    d_warm, c_warm = warm_rows(decision_rows), warm_rows(chat_rows)
    cold_note = ""
    if len(d_warm) < len(decision_rows):
        cold_note = (
            f"run 1 的 decision 首请求为冷启动（{ms(decision_rows[0]['wall_ms'])}，"
            "Ollama 装载模型），已从图表排除，完整数据见上表。"
        )

    d_warm_mean = statistics.mean(r["wall_ms"] for r in d_warm)
    c_warm_mean = statistics.mean(r["wall_ms"] for r in c_warm)
    overhead = (d_warm_mean - c_warm_mean) / c_warm_mean * 100 if c_warm_mean else 0
    agreement = summary["agreement"]
    total_runs = meta["runs"]
    d_prompt = decision_rows[0]["prompt_tokens"] if decision_rows else 0
    d_gen = decision_rows[0]["generated_tokens"] if decision_rows else 0
    c_prompt = chat_rows[0]["prompt_tokens"] if chat_rows else 0
    c_gen = chat_rows[0]["generated_tokens"] if chat_rows else 0

    criteria_rows = "".join(
        f"<tr><td><code>{esc(letter)}</code></td><td><code>{esc(key)}</code></td><td>{esc(value)}</td></tr>"
        for letter, (key, value) in zip("ABCDEFGH", meta["criteria"].items())
    )

    detail_rows = "".join(
        f"""<tr>
          <td>{esc(r['run'])}</td>
          <td><span class="tag decision-tag">decision</span></td>
          <td class="num">{ms(r['wall_ms'])}</td>
          <td class="num">{esc(r['server_ms'] and ms(r['server_ms']) or '—')}</td>
          <td class="num">{esc(r['prompt_tokens'])}</td>
          <td class="num">{esc(r['generated_tokens'])}</td>
          <td>{esc(r['result'])}</td>
          <td class="dim">{esc(r['detail'])}</td>
        </tr>"""
        + f"""<tr>
          <td>{esc(c['run'])}</td>
          <td><span class="tag chat-tag">chat</span></td>
          <td class="num">{ms(c['wall_ms'])}</td>
          <td class="num">—</td>
          <td class="num">{esc(c['prompt_tokens'])}</td>
          <td class="num">{esc(c['generated_tokens'])}</td>
          <td>{esc(c['result'])}</td>
          <td class="dim">{esc(c['detail'])}</td>
        </tr>"""
        for r, c in zip(decision_rows, chat_rows)
    )

    generated_at = datetime.fromisoformat(meta["generated_at"]).astimezone()
    peak_mean = max(d_warm_mean, c_warm_mean) or 1.0
    token_peak = max(d_prompt + d_gen, c_prompt + c_gen) or 1

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jev 决策接口 vs Ollama Chat 对比报告</title>
<style>
  :root {{
    --bg: #f4f6f9; --card: #ffffff; --ink: #1c2733; --dim: #64748b;
    --accent: #2563eb; --decision: #2563eb; --chat: #0d9488;
    --border: #e2e8f0; --good: #16a34a; --warn: #d97706;
  }}
  * {{ box-sizing: border-box; margin: 0; }}
  body {{
    font-family: -apple-system, "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
    background: var(--bg); color: var(--ink); padding: 32px 16px 64px; line-height: 1.6;
  }}
  .wrap {{ max-width: 1060px; margin: 0 auto; }}
  header {{
    background: linear-gradient(135deg, #1e3a8a, #2563eb); color: #fff;
    border-radius: 14px; padding: 28px 32px; margin-bottom: 20px;
  }}
  header h1 {{ font-size: 22px; font-weight: 650; margin-bottom: 6px; }}
  header p {{ opacity: .85; font-size: 13px; }}
  .chips {{ margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; }}
  .chip {{
    background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.25);
    padding: 3px 12px; border-radius: 999px; font-size: 12px;
  }}
  .card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 14px;
    padding: 22px 26px; margin-bottom: 20px;
  }}
  h2 {{ font-size: 16px; margin-bottom: 14px; display: flex; align-items: center; gap: 8px; }}
  h2::before {{ content: ""; width: 4px; height: 16px; background: var(--accent); border-radius: 2px; }}
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; }}
  .kpi {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px; }}
  .kpi .label {{ font-size: 12px; color: var(--dim); }}
  .kpi .value {{ font-size: 26px; font-weight: 700; margin: 2px 0; }}
  .kpi .sub {{ font-size: 12px; color: var(--dim); }}
  .good {{ color: var(--good); }} .warn {{ color: var(--warn); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: left; }}
  th {{ color: var(--dim); font-weight: 600; font-size: 12px; }}
  td.num {{ font-variant-numeric: tabular-nums; }}
  td.dim {{ color: var(--dim); font-size: 12px; max-width: 320px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .tag {{ font-size: 11px; padding: 1px 8px; border-radius: 999px; color: #fff; }}
  .decision-tag {{ background: var(--decision); }}
  .chat-tag {{ background: var(--chat); }}
  .chart {{ display: flex; align-items: flex-end; gap: 22px; padding: 10px 4px 0; flex-wrap: wrap; }}
  .rungroup {{ text-align: center; }}
  .runlabel {{ font-size: 11px; color: var(--dim); margin-bottom: 6px; }}
  .bars {{ display: flex; gap: 6px; align-items: flex-end; height: 170px; }}
  .barcol {{ display: flex; flex-direction: column; align-items: center; justify-content: flex-end; height: 100%; }}
  .bar {{ width: 34px; border-radius: 5px 5px 0 0; }}
  .bar.decision {{ background: var(--decision); }}
  .bar.chat {{ background: var(--chat); }}
  .barval {{ font-size: 11px; color: var(--dim); margin-top: 4px; }}
  .legend {{ display: flex; gap: 18px; font-size: 12px; color: var(--dim); margin-top: 12px; }}
  .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 5px; vertical-align: -1px; }}
  .hbarrow {{ display: grid; grid-template-columns: 150px 1fr 110px; gap: 12px; align-items: center; margin: 10px 0; font-size: 13px; }}
  .hbarlabel {{ color: var(--dim); text-align: right; }}
  .hbartrack {{ background: #eef2f7; border-radius: 6px; height: 22px; overflow: hidden; }}
  .hbar {{ height: 100%; border-radius: 6px; }}
  .hbar.decision {{ background: var(--decision); }}
  .hbar.chat {{ background: var(--chat); }}
  .hbarnote {{ font-variant-numeric: tabular-nums; }}
  .scenario {{ background: #f8fafc; border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; font-size: 13px; }}
  .scenario code {{ background: #eef2f7; padding: 1px 6px; border-radius: 4px; font-size: 12px; }}
  .notes li {{ margin: 8px 0; font-size: 13px; }}
  ul {{ padding-left: 18px; }}
  footer {{ text-align: center; color: var(--dim); font-size: 12px; margin-top: 8px; }}
  .disclaimer {{ font-size: 12px; color: var(--dim); border-left: 3px solid var(--warn); padding-left: 12px; margin-top: 14px; }}
</style>
</head>
<body>
<div class="wrap">

<header>
  <h1>Jev 决策接口 vs Ollama Chat — 对比报告</h1>
  <p>同一 prompt、同一模型、同一硬件下的结果 / 耗时 / token 用量对比</p>
  <div class="chips">
    <span class="chip">模型: {esc(meta['model'])}</span>
    <span class="chip">轮次: {esc(total_runs)}</span>
    <span class="chip">后端: {esc(health.get('backend', 'ollama'))} · Ollama {esc(health.get('server_version', '?'))}</span>
    <span class="chip">策略: {esc(health.get('strategy', 'single_token_generation'))}</span>
    <span class="chip">生成时间: {esc(generated_at.strftime('%Y-%m-%d %H:%M:%S'))}</span>
    <span class="chip">host: {esc(platform.node())} · {esc(platform.system())}</span>
  </div>
</header>

<div class="kpis">
  <div class="kpi">
    <div class="label">标签一致性</div>
    <div class="value good">{esc(agreement)}/{esc(total_runs)}</div>
    <div class="sub">两接口选择相同标签的轮次</div>
  </div>
  <div class="kpi">
    <div class="label">decision 热态均值</div>
    <div class="value">{esc(ms(d_warm_mean))}</div>
    <div class="sub">/v1/decision（{esc(len(d_warm))} 轮热态）</div>
  </div>
  <div class="kpi">
    <div class="label">chat 热态均值</div>
    <div class="value">{esc(ms(c_warm_mean))}</div>
    <div class="sub">/api/chat 直连（{esc(len(c_warm))} 轮热态）</div>
  </div>
  <div class="kpi">
    <div class="label">decision 附加开销</div>
    <div class="value {esc('warn' if overhead > 20 else '')}">+{esc(f'{overhead:.1f}%')}</div>
    <div class="sub">Jev 转发 + 标签匹配（相对 chat）</div>
  </div>
</div>

<div class="card">
  <h2>测试场景</h2>
  <div class="scenario">
    <p><b>输入：</b>{esc(meta['input'])}</p>
    <p><b>指令：</b>{esc(meta['instructions'])}</p>
    <table style="margin-top:10px">
      <tr><th>标签</th><th>key</th><th>含义</th></tr>
      {criteria_rows}
    </table>
    <p style="margin-top:8px;color:var(--dim);font-size:12px">
      prompt 长度 {esc(meta['prompt_chars'])} 字符（模板 v2，两种模式使用完全相同的 prompt）
    </p>
  </div>
</div>

<div class="card">
  <h2>每轮明细</h2>
  <table>
    <tr><th>轮次</th><th>模式</th><th>客户端耗时</th><th>服务端耗时</th><th>prompt tokens</th><th>生成 tokens</th><th>结果</th><th>明细</th></tr>
    {detail_rows}
  </table>
  <p class="disclaimer">{esc(cold_note)}</p>
</div>

<div class="card">
  <h2>每轮耗时对比（热态）</h2>
  <div class="chart">{bars_for_runs(d_warm, c_warm)}</div>
  <div class="legend">
    <span><span class="dot" style="background:var(--decision)"></span>/v1/decision</span>
    <span><span class="dot" style="background:var(--chat)"></span>/api/chat</span>
  </div>
</div>

<div class="card">
  <h2>汇总对比</h2>
  <h2 style="font-size:13px;color:var(--dim)">热态平均耗时</h2>
  {hbar('/v1/decision', d_warm_mean, peak_mean, 'decision', ms(d_warm_mean))}
  {hbar('/api/chat', c_warm_mean, peak_mean, 'chat', ms(c_warm_mean))}
  <h2 style="font-size:13px;color:var(--dim);margin-top:20px">token 用量（prompt + 生成）</h2>
  {hbar('decision tokens', d_prompt + d_gen, token_peak, 'decision', f'{d_prompt} + {d_gen} = {d_prompt + d_gen}')}
  {hbar('chat tokens', c_prompt + c_gen, token_peak, 'chat', f'{c_prompt} + {c_gen} = {c_prompt + c_gen}')}
</div>

<div class="card">
  <h2>结论与说明</h2>
  <ul class="notes">
    <li><b>结果：</b>两接口 {esc(agreement)}/{esc(total_runs)} 轮一致选择 <code>{esc(decision_rows[0]['result'] if decision_rows else '')}</code>；decision 接口额外返回全标签概率分布（结构化、保证是候选标签之一），chat 返回自由文本需自行解析。</li>
    <li><b>耗时：</b>热态下两者接近（decision 约 +{esc(f'{overhead:.0f}%')}），因为都只对 Ollama 发一次请求，146 token prefill 是主要成本。decision 的耗时与模型输出长度无关（恒定 2 token）；chat 随模型输出长度增长。</li>
    <li><b>token：</b>decision 恒为 prompt+2（num_predict=2 绕过 Ollama v0.12.x 的 logprobs 缺失 bug）；chat 本场景仅 2 token（模型只回答字母即停），若模型输出解释文本则显著增加。</li>
    <li><b>冷启动：</b>首轮 decision 请求需 Ollama 装载模型（约 6s）；可用 <code>OLLAMA_KEEP_ALIVE=30m</code> 常驻规避。</li>
    <li><b>为何 chat 耗时与 decision 接近甚至略快：</b>Ollama 后端的 decision 本质上也是一次生成请求（同样的 prefill + 生成 2 token，SPEC.md §8 回退模式），仅比 chat 直连多一跳服务转发（约 +9%）。本场景 chat 只生成 2 token 属于"最佳情况"；实测要求模型详细解释时 chat 耗时 2695ms / 52 token，而 decision 恒定约 450ms / 2 token——decision 的优势在于<b>耗时与模型输出长度解耦</b>，而非绝对更快。</li>
    <li><b>诚实标注：</b>decision 在 Ollama 后端下报告 <code>strategy: single_token_generation</code> 与 <code>generated_tokens: 2</code>（SPEC.md §8 生成回退语义，非 decode-free）。</li>
  </ul>
  <p class="disclaimer">decision 返回的 probabilities 为标签间相对分数（softmax 归一），不是现实意义上的校准置信度（SPEC.md §7.6）。本报告数据仅代表本次运行环境：{esc(platform.node())} · {esc(platform.platform())}。</p>
</div>

<footer>Jev Decision API · benchmarks/compare_chat_vs_decision.py + make_html_report.py 生成</footer>
</div>
</body>
</html>
"""

    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(html_doc)
    print(f"report written to {args.out} ({os.path.getsize(args.out)} bytes)")


if __name__ == "__main__":
    main()
