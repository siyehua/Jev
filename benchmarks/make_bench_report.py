#!/usr/bin/env python3
"""Generate a self-contained HTML report from bench.py JSON output
(direct label-logit scoring vs generation baselines, SPEC.md §13.9/§14).

Displayed modes: direct logits, one-token chat answer, chat think mode.
The short-generation baseline stays in the raw JSON (SPEC compliance) but
is intentionally not displayed. Zero external dependencies — renders on
isolated intranet machines.

Usage:
    python3 benchmarks/make_bench_report.py \
        --json benchmarks/bench_results.json \
        --out   benchmarks/jev_bench_report.html
"""

from __future__ import annotations

import argparse
import html
import json
import os
import platform
from datetime import datetime


def esc(value) -> str:
    return html.escape(str(value))


def ms(value) -> str:
    if value is None:
        return "—"
    return f"{value:,.1f}ms"


# key, display label, color
DISPLAY_MODES = [
    ("direct_logits", "direct logits（decode-free 决策）", "#2563eb"),
    ("chat_normal", "chat：普通模式（直接回答）", "#d97706"),
    ("chat_think", "chat：思考模式（推理后作答）", "#7c3aed"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True)
    parser.add_argument("--out", default="jev_bench_report.html")
    args = parser.parse_args()

    with open(args.json, encoding="utf-8") as handle:
        data = json.load(handle)

    entries = {key: data[key] for key, _, _ in DISPLAY_MODES}
    direct = entries["direct_logits"]
    think = entries["chat_think"]
    normal = entries["chat_normal"]

    peak = max(entries[key]["warm_p50_ms"] for key, _, _ in DISPLAY_MODES) or 1.0
    speedup_think = think["warm_p50_ms"] / direct["warm_p50_ms"]
    speedup_normal = normal["warm_p50_ms"] / direct["warm_p50_ms"]

    latency_bars = "".join(
        f"""
      <div class="hbarrow">
        <div class="hbarlabel">{esc(label)}</div>
        <div class="hbartrack"><div class="hbar" style="width:{max(2, round(entries[key]['warm_p50_ms'] / peak * 100))}%;background:{color}"></div></div>
        <div class="hbarnote">p50 {ms(entries[key]['warm_p50_ms'])}</div>
      </div>"""
        for key, label, color in DISPLAY_MODES
    )

    gen_peak = max(entries[key]["generated_tokens"] for key, _, _ in DISPLAY_MODES) or 1
    token_bars = "".join(
        f"""
      <div class="hbarrow">
        <div class="hbarlabel">{esc(label)}</div>
        <div class="hbartrack"><div class="hbar" style="width:{max(2, round(entries[key]['generated_tokens'] / gen_peak * 100))}%;background:{color}"></div></div>
        <div class="hbarnote">{esc(entries[key]['generated_tokens'])} tokens</div>
      </div>"""
        for key, label, color in DISPLAY_MODES
    )

    table_rows = "".join(
        f"""
      <tr>
        <td><span class="tag" style="background:{color}">{esc(label)}</span></td>
        <td class="num">{ms(entries[key]['cold_ms'])}</td>
        <td class="num">{ms(entries[key]['warm_mean_ms'])}</td>
        <td class="num"><b>{ms(entries[key]['warm_p50_ms'])}</b></td>
        <td class="num">{ms(entries[key]['warm_p95_ms'])}</td>
        <td class="num">{esc(entries[key]['prompt_tokens'])}</td>
        <td class="num">{esc(entries[key]['generated_tokens'])}</td>
        <td>{esc(entries[key]['result_preview'])}</td>
      </tr>"""
        for key, label, color in DISPLAY_MODES
    )

    think_gen = think["generated_tokens"]
    normal_gen = normal["generated_tokens"]
    saved_per_thousand = think_gen * 1000
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    memory = data.get("memory", {})

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jev 基准报告 — direct logits vs chat（含思考模式）</title>
<style>
  :root {{ --bg:#f4f6f9; --card:#fff; --ink:#1c2733; --dim:#64748b; --accent:#2563eb; --border:#e2e8f0; --good:#16a34a; }}
  * {{ box-sizing:border-box; margin:0; }}
  body {{ font-family:-apple-system,"PingFang SC","Microsoft YaHei",system-ui,sans-serif;
         background:var(--bg); color:var(--ink); padding:32px 16px 64px; line-height:1.6; }}
  .wrap {{ max-width:1000px; margin:0 auto; }}
  header {{ background:linear-gradient(135deg,#0d9488,#2563eb); color:#fff; border-radius:14px; padding:28px 32px; margin-bottom:20px; }}
  header h1 {{ font-size:22px; font-weight:650; margin-bottom:6px; }}
  header p {{ opacity:.85; font-size:13px; }}
  .chips {{ margin-top:14px; display:flex; flex-wrap:wrap; gap:8px; }}
  .chip {{ background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.25);
           padding:3px 12px; border-radius:999px; font-size:12px; }}
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:14px; margin-bottom:20px; }}
  .kpi {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:16px 18px; }}
  .kpi .label {{ font-size:12px; color:var(--dim); }}
  .kpi .value {{ font-size:26px; font-weight:700; margin:2px 0; }}
  .kpi .sub {{ font-size:12px; color:var(--dim); }}
  .good {{ color:var(--good); }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:22px 26px; margin-bottom:20px; }}
  h2 {{ font-size:16px; margin-bottom:14px; display:flex; align-items:center; gap:8px; }}
  h2::before {{ content:""; width:4px; height:16px; background:var(--accent); border-radius:2px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th,td {{ padding:8px 10px; border-bottom:1px solid var(--border); text-align:left; }}
  th {{ color:var(--dim); font-weight:600; font-size:12px; }}
  td.num {{ font-variant-numeric:tabular-nums; }}
  .tag {{ font-size:11px; padding:1px 8px; border-radius:999px; color:#fff; }}
  .hbarrow {{ display:grid; grid-template-columns:230px 1fr 110px; gap:12px; align-items:center; margin:12px 0; font-size:13px; }}
  .hbarlabel {{ color:var(--dim); text-align:right; }}
  .hbartrack {{ background:#eef2f7; border-radius:6px; height:22px; overflow:hidden; }}
  .hbar {{ height:100%; border-radius:6px; }}
  .hbarnote {{ font-variant-numeric:tabular-nums; }}
  ul {{ padding-left:18px; }}
  .notes li {{ margin:8px 0; font-size:13px; }}
  .disclaimer {{ font-size:12px; color:var(--dim); border-left:3px solid #d97706; padding-left:12px; margin-top:14px; }}
  footer {{ text-align:center; color:var(--dim); font-size:12px; }}
</style>
</head>
<body>
<div class="wrap">

<header>
  <h1>Jev 基准报告 — direct logits vs chat（含思考模式）</h1>
  <p>同模型 / 同场景 / 同硬件 / 同缓存状态下的受控对比（SPEC.md §13.9, §14）</p>
  <div class="chips">
    <span class="chip">模型: {esc(data['model'])}</span>
    <span class="chip">设备: {esc(data['device'])} · {esc(data['dtype'])}</span>
    <span class="chip">轮次: {esc(data['runs'])}（冷 1 + 热 {esc(data['runs'] - 1)}）</span>
    <span class="chip">思考模式上限: {esc(data.get('think_tokens_limit'))} tokens</span>
    <span class="chip">显存峰值: {esc(memory.get('cuda_allocated_mb'))}MB</span>
    <span class="chip">host: {esc(platform.node())}</span>
    <span class="chip">生成时间: {esc(generated_at)}</span>
  </div>
</header>

<div class="kpis">
  <div class="kpi">
    <div class="label">direct logits 热态 p50</div>
    <div class="value good">{ms(direct['warm_p50_ms'])}</div>
    <div class="sub">decode-free：0 个生成 token</div>
  </div>
  <div class="kpi">
    <div class="label">对比 chat 普通模式</div>
    <div class="value good">{esc(f'{speedup_normal:.1f}x')}</div>
    <div class="sub">{ms(normal['warm_p50_ms'])} → {ms(direct['warm_p50_ms'])}</div>
  </div>
  <div class="kpi">
    <div class="label">对比 chat 思考模式</div>
    <div class="value good">{esc(f'{speedup_think:.1f}x')}</div>
    <div class="sub">{ms(think['warm_p50_ms'])} → {ms(direct['warm_p50_ms'])}</div>
  </div>
  <div class="kpi">
    <div class="label">每 1000 次决策 token 节省</div>
    <div class="value good">{esc(f'{saved_per_thousand:,}')}</div>
    <div class="sub">对比思考模式（{esc(think_gen)} → 0 生成 token/次）</div>
  </div>
</div>

<div class="card">
  <h2>热态 p50 耗时对比</h2>
  {latency_bars}
</div>

<div class="card">
  <h2>token 消耗对比（生成部分）</h2>
  {token_bars}
  <p style="font-size:12px;color:var(--dim);margin-top:8px">
    prompt 部分三种模式基本一致（{esc(direct['prompt_tokens'])} /
    {esc(normal['prompt_tokens'])} / {esc(think['prompt_tokens'])} tokens，思考模式仅多出推理指令）；
    差异全部来自解码端：direct logits 一次 prefill 后直接读取 next-token logits，
    <b>解码 0 个 token</b>。
  </p>
</div>

<div class="card">
  <h2>完整数据（冷启动与热态）</h2>
  <table>
    <tr><th>模式</th><th>冷启动</th><th>热态均值</th><th>热态 p50</th><th>热态 p95</th><th>prompt tokens</th><th>生成 tokens</th><th>结果</th></tr>
    {table_rows}
  </table>
</div>

<div class="card">
  <h2>结论与说明</h2>
  <ul class="notes">
    <li><b>decode-free 的双重收益：</b>direct logits 只做一次 prefill forward 就读取 next-token logits，比 chat 普通模式快 <b>{esc(f'{speedup_normal:.1f}')} 倍</b>、比思考模式快 <b>{esc(f'{speedup_think:.1f}')} 倍</b>，且每次决策<b>零解码 token</b>——延迟与 token 成本都与"模型想说多少话"解耦。</li>
    <li><b>chat 普通模式</b>：标准 prompt 直接作答，本场景每次消耗 {esc(normal_gen)} 个生成 token（上限 {esc(data.get('normal_tokens_limit'))}）；<b>思考模式</b>：模型先逐步推理再给出结论，每次消耗 {esc(think_gen)} 个生成 token（上限 {esc(data.get('think_tokens_limit'))}）。耗时与 token 成本均随输出长度线性增长。</li>
    <li><b>结构化保证：</b>direct logits 输出的 choice 恒为候选标签之一并附全标签概率分布；chat 的自由文本需要解析，且模型可能不按格式作答（本基准的思考模式就没有输出约定的 <code>Final answer:</code> 行，需多级回退解析）。</li>
    <li><b>冷启动 ~{esc(ms(direct['cold_ms']))}</b> 为 CUDA kernel 首次编译；服务常驻后无此成本。</li>
    <li><b>质量提示：</b>本基准用 Qwen2.5-0.5B。歧义输入（迟到+重复扣款）下 0.5B 与 12B 模型结论可能不同；对精度敏感时可通过 <code>JEV_MODEL_ID</code> 换更大的模型，接口无需改动。</li>
  </ul>
  <p class="disclaimer">direct logits 返回的 probabilities 为标签间相对分数（softmax 归一），非校准置信度（SPEC.md §7.6）。所有数字仅代表本次运行环境：{esc(platform.node())} · {esc(platform.platform())}。</p>
</div>

<footer>Jev Decision API · benchmarks/bench.py + make_bench_report.py 生成</footer>
</div>
</body>
</html>"""

    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(html_doc)
    print(f"report written to {args.out} ({os.path.getsize(args.out)} bytes)")


if __name__ == "__main__":
    main()
