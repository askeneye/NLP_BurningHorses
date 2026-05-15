# Ideas for visualisations

#### Performance vs K-shot

X-axis: K-shot values

Y-axis: Strict Span-F1

Lines:

* BERT baseline
* PET
* OADA
* PET + OADA

#### Relative gain over baseline

Δ Strict Span F1 over vanilla BERT

Improvements should become visually clear. Example below:

| K  | BERT | PET+OADA | Gain |
| -- | ---- | -------- | ---- |
| 5  | 42   | 55       | +13  |
| 50 | 78   | 81       | +3   |

#### OADA permutation visualization

Sentence:

<pre class="overflow-visible! px-0!" data-start="1510" data-end="1548"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="relative"><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><pre class="cm-content q9tKkq_readonly m-0"><code><span>Barack Obama visited Paris</span></code></pre></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

Original extraction:

<pre class="overflow-visible! px-0!" data-start="1572" data-end="1612"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="relative"><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><pre class="cm-content q9tKkq_readonly m-0"><code><span>[Barack Obama]PER [Paris]LOC</span></code></pre></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

OADA permutations:

<pre class="overflow-visible! px-0!" data-start="1634" data-end="1738"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="relative"><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><pre class="cm-content q9tKkq_readonly m-0"><code><span>Order: PER LOC</span><br/><span>→ [Barack Obama]PER [Paris]LOC</span><br/><br/><span>Order: LOC PER</span><br/><span>→ [Paris]LOC [Barack Obama]PER</span></code></pre></div></div></div></div></div></div></div></div></div></div></div></div></pre>

#### PET teacher → lightweight student

Input sentence
      ↓
Pattern / Prompt
      ↓
Generative Teacher Ensemble
      ↓
Soft pseudo-labels
      ↓
Distillation
      ↓
Lightweight BERT student
      ↓
Parallel token tagging

#### Rare-class improvement

X-axis:

* number of training examples per entity type

Y-axis:

* F1 improvement over baseline
