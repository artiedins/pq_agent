# Essential Elements of AI Intelligence Benchmark Bar Charts

When designing bar charts to communicate AI intelligence benchmark results, the industry relies on a standardized visual grammar so readers can parse dense model lineups in seconds. 

---

## 1. Things That Are Very Common (Industry Standards)

* **Lab-Specific Color Coding:** Rather than coloring bars by score, charts assign a consistent "house color" to each AI lab so readers can visually cluster who is leading:
  * **Anthropic (Claude):** Warm terracotta, burnt orange, or coral.
  * **OpenAI (GPT / o-series):** Muted teal, dark slate, or charcoal black.
  * **Google (Gemini):** Royal blue or periwinkle.
  * **xAI (Grok):** Stark monochrome black/white or dark gray.
  * **DeepSeek / Meta / Qwen:** Distinct blues, purples, or bright indigos.
* **Strictly Descending Sort Order:** Bars are always ordered from highest score to lowest score (top-to-bottom on horizontal charts, left-to-right on vertical charts) to establish an instant visual hierarchy.
* **Lab Logos + Stripped Model Names:** Company logos sit directly beside or beneath the bar, allowing labels to drop redundant company prefixes (e.g., showing `[Anthropic Logo] Opus 5.5` instead of `Anthropic Claude Opus 5.5`).
* **Direct Data Labels:** Exact numeric scores are printed directly at the tip of every bar (e.g., `68.4` or `16.4%`) so readers do not have to trace eye-lines back to an axis.
* **Zero-Baseline Axis:** The numeric axis starts at `0` so bar lengths remain strictly proportional to actual scores.
* **Versioned Context Header:** Titles explicitly state the benchmark name, version number, and date (e.g., *Intelligence Index v5.0 — Nov 2026*) so screenshots remain accurate as eval suites update.

---

## 2. Things to Potentially Consider (Helpful Enhancements)

* **Reasoning Effort Tags & Shade Variations:** Because frontier models scale performance via test-time compute, consider appending thinking budgets in parentheses—such as `(High)`, `(xhigh)`, `(Max)`, or `(No Thinking)`—and using lighter or darker shades of the same lab color to separate reasoning variants from base models.
* **Open-Weight vs. Closed Signifiers:** If mixing proprietary APIs and open-weight models, consider distinguishing them via solid vs. striped/hatched bar fills, an `Open` text badge, or a legend marker.
* **Horizontal Orientation for Large Lineups:** While vertical bars work well for ≤10 models, consider horizontal bars when displaying 12+ models so long names with reasoning tags stay readable without rotated text.
* **95% Confidence Intervals (Error Bars):** Adding subtle error whiskers at the tips of bars helps readers avoid over-interpreting statistically noisy 0.3-point gaps between adjacent models.
* **Color Grouping for Minor Labs:** If your chart features more than 5 or 6 organizations, consider assigning signature colors only to the top labs and grouping remaining entrants under a neutral gray to prevent visual clutter.
