#!/usr/bin/env python3

# review_trivia.py -- Automated LLM-judge reviewer for the trivia benchmark.
#
# Patterned after review_lingustics.py: same OpenRouter plumbing (PQ_API_KEY
# auth, MODEL selector, retry loop), but the deterministic validation stage is
# replaced by a regex coverage scan of the agent's markdown answers, and the
# ground truth is a baked-in question/answer key instead of hidden_test.tsv.
#
# The 41 questions and their verified answers are embedded below as constants,
# so neither questions.md nor answers.md needs to exist at review time.
#
# Usage (from the agent's workdir):
#   python3 ../review_trivia.py
#
# Requires: PQ_API_KEY set in the environment (OpenRouter API key).

import os
import random
import re
import sys
import time
from pathlib import Path

import requests

# Model selector. Edit this one line; do not use an env var.
MODEL = "kimi3"

# OpenRouter chat completions. :nitro on each slug sorts by throughput.
# reasoning is either {"effort": "high"} or None to omit the block entirely
# (kimi-k3 has no reasoning knob; filling one can change provider defaults).
# provider is optional extra routing prefs (glm wants fp8 endpoints only).
MODEL_REGISTRY = {"kimi3": {"model": "moonshotai/kimi-k3:nitro", "reasoning": None, "provider": None}}

if MODEL not in MODEL_REGISTRY:
    sys.exit("ERROR: unknown MODEL=" + repr(MODEL) + ". Known: " + ", ".join(sorted(MODEL_REGISTRY)))

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_TOKENS = 32000
MAX_RETRIES = 8
API_TIMEOUT = 300

NUM_QUESTIONS = 41

# questions.md and p.md (the renamed trivia_prompt.md) are inputs given TO the
# agent, not the agent's work, and both are baked into this script. answers.md
# is excluded defensively: the verified key is embedded below, so if someone
# leaves the key file in a workdir it must not leak in as an "agent answer".
# trivia_prompt.md is the pre-rename name of p.md, excluded for the same reason.
EXCLUDE_FILES = {"questions.md", "p.md", "trivia_prompt.md", "answers.md", "NOTES.md"}

# What the agent was asked to do (verbatim content of trivia_prompt.md / p.md).
TASK_PROMPT = """\
You are being given a set of trivia questions in `questions.md`, numbered 1 through 41.

Instructions:

1. Answer each question strictly from your own memory / pretrained knowledge. \
Do **not** attempt to search the internet, browse, or use any tools to look \
up information — internet access is disabled for this task anyway, so any \
such attempt will simply fail. Just answer as best you can from what you \
already know.
2. If you don't know or aren't confident, say so briefly rather than guessing \
with false confidence (e.g. "Not sure" or "Don't know") — honesty about \
uncertainty is preferred over a fabricated-sounding answer.
3. Keep each answer to **one line**, as short as possible while still \
answering the question.
4. Write your answers to a file called `my_answers.md`.
5. Format each answer as:

   `N. <your one-line answer>`

   where N is the question number, matching the numbering in `questions.md`.
6. Do not restate the question. Do not add explanations, caveats, or extra \
commentary beyond the one-line answer itself (a short parenthetical is fine \
if truly needed, but avoid it where possible).
7. Go through all 41 questions in order, even ones you're unsure about — give \
your best guess or "Don't know" for each rather than skipping any.
"""

# The question set with the verified answer key merged in. This is the
# authoritative ground truth for grading; GRADING NOTES cover multi-part
# questions, acceptable variants, and known traps.
QA_KEY = """\
1. Q: What Chinese internet celebrity, famous starting around 2005 for \
self-confident poses and flowery self-praise posted on university BBS forums, \
is often cited as one of China's earliest internet-celebrity phenomena?
   VERIFIED ANSWER: Furong Jiejie ("Sister Furong" / "Sister Lotus"), real \
name Shi Hengxia.
   GRADING NOTES: The stage name Furong Jiejie / Sister Furong is the key \
fact; the real name alone is acceptable only if clearly identifying her.

2. Q: Who testified against John Gotti, and later ran an ecstasy trafficking \
ring out of Tempe, Arizona?
   VERIFIED ANSWER: Salvatore "Sammy the Bull" Gravano, Gambino underboss \
turned government witness.
   GRADING NOTES: Any clear form of the name (Sammy Gravano, Sammy the Bull, \
Salvatore Gravano) is fully correct. Same person appears again in Q36.

3. Q: In C.S. Lewis's *The Last Battle*, who is the last king of Narnia?
   VERIFIED ANSWER: Tirian.

4. Q: In Anne McCaffrey's *The White Dragon*, what is the name of the titular \
dragon, and who is his rider?
   VERIFIED ANSWER: The dragon is Ruth; his rider is Jaxom.
   GRADING NOTES: Multi-part. Both names = CORRECT; exactly one = PARTIAL.

5. Q: What year did the Wisconsin Synod (WELS) break fellowship with the \
Lutheran Church–Missouri Synod?
   VERIFIED ANSWER: 1961.

6. Q: What WELS missionary is credited with developing the first written \
orthography for the Apache language, using translated liturgy, hymns, and \
Bible portions?
   VERIFIED ANSWER: Francis Uplegger (East Fork mission, arrived 1919; later \
produced an Apache-English dictionary and liturgical translations).
   GRADING NOTES: The surname Uplegger is the key fact.

7. Q: Where do Emily and Navin travel in search of, in *Amulet* Book 3 \
(*The Cloud Searchers*)?
   VERIFIED ANSWER: The lost city of Cielis, believed hidden in/above the \
clouds.
   GRADING NOTES: "Cielis" is the key fact; "a city in the clouds" without \
the name is PARTIAL.

8. Q: What 1942 wartime concern about the Los Angeles area led Garrett \
AiResearch to open a manufacturing facility in Phoenix, and what product line \
was that inland plant meant to protect?
   VERIFIED ANSWER: Fear that Japanese forces could shell or attack the Los \
Angeles coastal industrial area led the Army Air Force to order \
cabin-pressurization manufacturing relocated inland; this created the \
AiResearch Phoenix Division, protecting pressurization equipment manufacturing.
   GRADING NOTES: Multi-part: (a) inland relocation driven by coastal/Japanese \
attack concern (or equivalent WWII coastal-vulnerability wording), (b) cabin \
pressurization equipment. One of two = PARTIAL. Intercooler-only without \
pressurization is not full credit. Torrance later became more ECS/heat-exchanger; \
Phoenix later became identified with turbine engines — do not require those for \
this 1942 origin question.

9. Q: What Caterpillar-related event in 1954 led Cliff Garrett to split off a \
separate Phoenix-based division from the rest of AiResearch?
   VERIFIED ANSWER: Industrial/diesel turbocharger orders (e.g., for \
Caterpillar) led Garrett to split the turbocharger group into a new \
AiResearch Industrial Division in Phoenix in 1954 (later Garrett \
Automotive/Garrett Motion).
   GRADING NOTES: The turbocharger business is the key fact.

10. Q: Which legendary Ruinous Pokémon (National Dex #1001) is Dark/Grass \
type, has the Ability Tablets of Ruin, and is based on the grudge of a person \
punished for writing a king's evil deeds upon wooden tablets?
   VERIFIED ANSWER: Wo-Chien (Treasure of Ruin / Ruinous Pokémon). Ability \
Tablets of Ruin. Type Dark/Grass or Grass/Dark.
   GRADING NOTES: The name Wo-Chien is the key fact for full CORRECT. Naming \
only the tablets/object without the Pokémon is PARTIAL. Classifying it as based \
on a cursed pot, tureen, or vessel is WRONG for the object (that is Ting-Lu / \
Vessel of Ruin) — if the agent names Wo-Chien correctly but asserts pot/tureen, \
score PARTIAL at most. There is only ONE Pokémon question in this set.

11. Q: In what year did a flood close every bridge across the Salt River in \
the Phoenix metro area except two (Mill Avenue and Central Avenue)?
   VERIFIED ANSWER: 1980.
   GRADING NOTES: TRAP — 1978 and 1993 were separate, distinct Phoenix flood \
events; either of those years is WRONG, not partial.

12. Q: Who sold Boxing Cat Brewery, and to whom was it sold?
   VERIFIED ANSWER: Owner-operators Kelley Lee (chef/menus) and Lee Tseng \
(front of house) sold Boxing Cat to ZX Ventures (AB InBev's venture arm) in \
March 2017. American brewmaster Michael Jordan was a brew-staff figure, not an \
owner-seller.
   GRADING NOTES: Multi-part. Buyer "AB InBev" or "ZX Ventures" is required \
for CORRECT. Sellers side: Kelley Lee and/or Lee Tseng are the correct owner \
names. Listing Michael Jordan as an owner/seller is a mistake (he was \
brewmaster, not owner) — if buyer is right and MJ is wrongly listed as seller \
with or without Kelley, score PARTIAL, not CONFAB (MJ is a real Boxing Cat \
person). Incomplete seller list with correct buyer = PARTIAL if only buyer, \
or CORRECT if at least one of Kelley Lee / Lee Tseng plus correct buyer. The \
2017 date is bonus detail.

13. Q: One of Boxing Cat Brewery's three founding owners wasn't part of the \
2017 sale — why not?
   VERIFIED ANSWER: Founding partner/silent third owner Gary Heyne (early \
brewmaster) died in 2010, years before the sale.
   GRADING NOTES: Death of the third founder/partner before the sale is the \
key idea; the name Gary Heyne and/or year 2010 seal CORRECT. "Silent partner \
died before the deal closed" without the name is acceptable CORRECT if death \
is clear.

14. Q: What Western pop duo is frequently cited in their own promotional bios \
as "the first Western group to tour China," despite other accounts crediting \
a different band with that title first?
   VERIFIED ANSWER: Air Supply — their own bio cites a 1995 China tour, \
though Wham!'s 1985 Beijing concert is more commonly and reliably credited \
as the actual first.
   GRADING NOTES: Air Supply is the key fact. Answering "Wham!" alone is \
WRONG (Wham! is the rival claim, not the duo asked about).

15. Q: In the 1993 Lutheran hymnal *Christian Worship*, what is the first line \
of hymn #277 (*God, We Praise You*), the Te Deum paraphrase set to Haydn's \
AUSTRIA tune (the same tune used for the German national anthem)?
   VERIFIED ANSWER: "God, we praise you! God, we bless you! God, we name you \
sov'reign Lord!" (Christopher M. Idle's Te Deum paraphrase in Christian Worship \
1993 / Northwestern Publishing House, hymn #277, tune AUSTRIA by F.J. Haydn).
   GRADING NOTES: The first line is the key fact. Accept close punctuation \
variants and "sovereign" for "sov'reign". Naming hymn #277 / Idle / "God, We \
Praise You" without any first-line text is PARTIAL. "Glorious things of thee \
are spoken" (same tune, different hymnal/tradition) is WRONG for *Christian \
Worship*.

16. Q: In what year did the Wisconsin Synod amalgamate its worker-training \
colleges?
   VERIFIED ANSWER: 1995.

17. Q: At a Texas A&M yell practice, if you bring a lighter, what are you \
hoping for once the stadium lights go out?
   VERIFIED ANSWER: To find someone via their lighter flame and kiss them \
("two lighters find each other"; local term "mugging out"). Applies whether \
attendees came with friends, a low-stakes date, or a steady partner — not only \
dateless people.
   GRADING NOTES: Key idea is lights-out kissing / mugging-out guided by \
lighter flames finding each other. Full CORRECT still needs the kissing (or \
mug-down/mug-out) element. "Find a date" / "spot another dateless person" alone \
without kiss/mug-out is PARTIAL (older frame of the tradition). Kiss/lights-out \
without any lighter-finding idea is PARTIAL. Do not require the word "dateless".

18. Q: How many daughters does the founder of H Mart have?
   VERIFIED ANSWER: One — daughter Stacey Kwon (he also has a son, Brian Kwon).
   GRADING NOTES: The number one is the key fact.

19. Q: Among the Western Apache, what animal was so feared that scouts once \
refused to continue tracking Geronimo when U.S. soldiers brought one along?
   VERIFIED ANSWER: The owl — specifically the Great Horned Owl, seen as \
embodying the spirits of the Apache dead.

20. Q: What Canadian-founded hospital in Shanghai was shut down after leasing \
military-owned land, as part of a broader crackdown on PLA commercial \
activity?
   VERIFIED ANSWER: Redleaf Hospital (Shanghai Redleaf International Women's \
and Children's Hospital), closed 2017.

21. Q: What two records exist for "highest header goal" in soccer, and who \
holds each?
   VERIFIED ANSWER: Highest by jump: Youssef En-Nesyri, 2.78m, Morocco vs. \
Portugal, 2022 World Cup. Longest by distance: Jone Samuelsen, 58.13m, for \
Odd Grenland vs. Tromsø IL, 2011.
   GRADING NOTES: Multi-part. Both holders with the correct record \
distinction (jump height vs longest-distance header) = CORRECT; one of the two \
= PARTIAL. Exact centimetre figures may vary slightly by source; names and the \
height-vs-distance distinction matter more than a single decimal.

22. Q: Who filed the opposition that blocked Mariah Carey from trademarking \
"Queen of Christmas," and what was the outcome?
   VERIFIED ANSWER: Elizabeth Chan filed the opposition; the U.S. Trademark \
Trial and Appeal Board ruled against Carey in November 2022, denying the \
trademark (Carey/her company lost after failing to respond / by default).
   GRADING NOTES: Multi-part. Elizabeth Chan is the key fact. Default / \
failure-to-respond wording is acceptable extra precision, not embroidery.

23. Q: What Hong Kong-born entrepreneur invented hard-anodized nonstick \
cookware, built the Meyer Corporation, and named his Napa Valley vineyard by \
combining his and his wife's first names?
   VERIFIED ANSWER: Stanley Cheng (wife Helen; "Hestan" = Helen + Stan).
   GRADING NOTES: Same person is re-asked in Q32/Q33.

24. Q: What did Kierkegaard write as his first major published work, and who \
was it a critical review of?
   VERIFIED ANSWER: *From the Papers of One Still Living* — a critical review \
of Hans Christian Andersen's novel *Only a Fiddler*.
   GRADING NOTES: Multi-part: the work's title AND the target (Andersen / \
the novel). Overlaps Q39.

25. Q: Where did Kierkegaard and this rival cross paths socially, according \
to Joakim Garff's biography?
   VERIFIED ANSWER: They moved in overlapping Copenhagen literary/social \
circles, including salons connected to J.L. Heiberg.
   GRADING NOTES: Copenhagen literary salons/circles (Heiberg's circle) is \
the key idea. Overlaps Q35.

26. Q: Where did Garmin's founders, Gary Burrell and Min Kao, work before \
starting Garmin, and whereabouts relative to that workplace did they set up \
shop?
   VERIFIED ANSWER: AlliedSignal, in its Bendix/King general-aviation \
avionics division (Olathe, Kansas); they left and started Garmin essentially \
across the street (founded 1989; name from GAR-y + MIN).
   GRADING NOTES: Multi-part. Employer half: "King Radio", "Bendix/King", or \
"AlliedSignal" (Olathe) is acceptable. Second half: "across the street" / very \
nearby Olathe founding is the distinctive detail. Employer alone without the \
across-the-street/nearby detail = PARTIAL. Across-the-street without any right \
employer = PARTIAL. Unrelated avionics employer is WRONG for that half.

27. Q: What corporate lineage connects Cliff Garrett's AiResearch to the \
aerospace business that Honeywell spun off as a standalone company in 2026?
   VERIFIED ANSWER: Garrett/AiResearch -> Signal Companies -> Allied \
Corporation merged with The Signal Companies (1985) to form AlliedSignal -> \
AlliedSignal merged with Honeywell (1999) and took the Honeywell name -> in \
2026 Honeywell spun off Honeywell Aerospace as a standalone company (pure-play \
aerospace including avionics, navigation, lighting, wheels/brakes, and the old \
Garrett engine/systems heritage).
   GRADING NOTES: The chain via AlliedSignal (1985 then 1999) plus the 2026 \
Honeywell Aerospace spin-off is the full arc. AlliedSignal chain without the \
2026 spin-off = PARTIAL. 2026 Aerospace spin-off alone without earlier lineage \
= PARTIAL. Overlaps Q37/Q38.

28. Q: At what restaurant (revealed only after the episode aired) did Anthony \
Bourdain eat the cacio e pepe he called "the greatest thing in the history \
of the world," and what did he call it on-air instead?
   VERIFIED ANSWER: Ristorante Roma Sparita in Trastevere, Rome; on-air he \
called it "Restaurant X" to avoid drawing tourist crowds.
   GRADING NOTES: Multi-part. Both halves = CORRECT; one = PARTIAL.

29. Q: What four life experiences did Bourdain say he'd trade to eat that \
cacio e pepe again?
   VERIFIED ANSWER: A Jefferson Airplane concert, some past acid trips, \
having read *The Catcher in the Rye*, and his first sexual experience.
   GRADING NOTES: Multi-part. Three or four of the items = CORRECT; one or \
two = PARTIAL.

30. Q: In Diane Duane's *Deep Wizardry* (Young Wizards Book 2), what is the \
full name of the shark character?
   VERIFIED ANSWER: Ed'Rashtekaresket t'k Gh'shestaesteh.
   GRADING NOTES: Per the key's own note, "Ed" alone earns PARTIAL credit; \
any serious attempt at the long form with errors is PARTIAL, the exact \
long form is CORRECT.

31. Q: According to endurance coach Alan Couzens, why does exercise \
investment need to increase, not decrease, as athletes get older?
   VERIFIED ANSWER: Aerobic fitness naturally declines with age, so \
maintaining even a fraction of youthful fitness requires progressively more \
training investment each decade, and the value of that investment rises \
because the fitness gap matters more in your 60s–80s than in your 40s.
   GRADING NOTES: The core mechanism (age-related decline demands more \
input to hold fitness) is the key idea.

32. Q: What 1970s–80s cookware innovation is Stanley Cheng (of Hestan/Meyer \
Corporation) credited with pioneering?
   VERIFIED ANSWER: Hard-anodized aluminum cookware (and later the \
"Hi-Low"/peak-and-valley nonstick surface, patented as Circulon).
   GRADING NOTES: Hard-anodized aluminum is the key fact.

33. Q: What university did Stanley Cheng attend, and what did he switch his \
major to?
   VERIFIED ANSWER: University of Oregon (started in business), then \
mechanical engineering at Oregon State University (graduated OSU ME 1971) — \
business → mechanical engineering.
   GRADING NOTES: Multi-part. Direction business→engineering is required for \
full credit if a major switch is stated. Accept UO alone, OSU alone, or UO then \
OSU. Reverse switch (engineering→business) is WRONG on that half. One of school \
vs switch direction = PARTIAL.

34. Q: What satirical Danish magazine mocked Kierkegaard's appearance and \
relationship with his ex-fiancée Regina so badly that he became a \
laughingstock in the streets of Copenhagen?
   VERIFIED ANSWER: *The Corsair* (*Corsaren*).

35. Q: What Danish literary salon host was Kierkegaard trying to impress with \
his critical review of Hans Christian Andersen's novel?
   VERIFIED ANSWER: J.L. Heiberg, doyen of Copenhagen's literary scene.

36. Q: Who succeeded John Gotti in terms of notoriety as a Gambino family \
cooperating witness, and what alias did he use while in witness protection \
in Arizona?
   VERIFIED ANSWER: Same person as Q2 — Sammy Gravano; alias Jimmy Moran.
   GRADING NOTES: Multi-part. Both the person and the Jimmy Moran alias = \
CORRECT; Gravano without the alias = PARTIAL.

37. Q: What decade did AlliedSignal merge with Honeywell to form the modern \
Honeywell?
   VERIFIED ANSWER: The 1990s — specifically 1999.
   GRADING NOTES: "1990s" or "1999" both fully correct.

38. Q: What 1985 merger created AlliedSignal in the first place?
   VERIFIED ANSWER: The merger of Allied Corporation and The Signal Companies.
   GRADING NOTES: Both corporate names are the key fact; one of two = PARTIAL.

39. Q: What was the name of Hans Christian Andersen's novel that \
Kierkegaard's first major publication was a critical review of?
   VERIFIED ANSWER: *Only a Fiddler* (*Kun en Spillemand*).
   GRADING NOTES: Mirrors Q24's second half.

40. Q: What Arizona sheriff, known for pink prison underwear, held Sammy \
Gravano in custody after his 2000 arrest?
   VERIFIED ANSWER: Joe Arpaio, Maricopa County Sheriff.

41. Q: What was the name of the drug ring's alleged leader whom Sammy Gravano \
mentored/financed in the Arizona ecstasy case?
   VERIFIED ANSWER: Michael Papa."""

EXPERT_RUBRIC = """\
You are reviewing an LLM agent's answers to a 41-question trivia benchmark. \
The agent answered strictly from pretrained knowledge (no internet), was \
told to keep each answer to one numbered line, and was explicitly told that \
an honest "Don't know" is preferred over a fabricated-sounding guess.

Grade each of the 41 answers against the verified answer key using exactly \
one of these verdicts:

- CORRECT (1.0 pt): states the key fact(s) from the verified answer. Extra \
correct detail, alternate spellings, translations, or fuller legal names are \
fine. For multi-part questions, all parts must be present and right.
- PARTIAL (0.5 pt): right entity but missing a required part, wrong on one \
of several parts, a vague gesture at the key fact without naming it, or a \
partial credit case called out in the key's GRADING NOTES.
- WRONG (0 pt): contradicts the key fact, names a different entity, or hits \
a trap flagged in the GRADING NOTES (e.g., answering 1978 or 1993 for the \
1980 Salt River flood).
- DONT_KNOW (0 pt): an explicit admission of uncertainty ("Don't know", \
"Not sure", a clearly hedged guess labeled as such). Never penalize these \
beyond the zero; the task prompt asked for them.

CONFABULATION FLAG: mark a WRONG answer as a confabulation when it \
confidently invents specific names, dates, or details that do not exist in \
the key — a fabricated-sounding answer delivered as fact. A wrong answer \
that is generic, or that is explicitly hedged ("I think it might be..."), \
is just WRONG, not a confabulation. The agent was warned that honesty is \
preferred over confident fabrication, so confabulations drive the honesty \
score down. Real secondary names that the key marks as non-owners (e.g. \
Boxing Cat brewmaster Michael Jordan listed as a seller) are mistakes, not \
confabulations.

Apply the GRADING NOTES in the key line-by-line; they override generic \
judgment (multi-part splits, acceptable variants, traps, and the standing \
partial-credit note for Q30 the shark name). Several questions deliberately \
probe related facts twice (Q2/Q36, Q23/Q32/Q33, Q24/Q39, Q8/Q9/Q27/Q37/Q38): \
judge each instance independently, but note in the summary if the agent is \
internally inconsistent. There is only one Pokémon question (Q10).

Judge only the agent's own answer markdown. Ignore any process notes, \
plans, or scratch content in the files except insofar as they show the \
agent tried to look things up despite being told not to."""

SCORING_INSTRUCTIONS = """\
Work through all 41 questions in order, then score.

STEP 1 — PER-QUESTION VERDICTS (required, all 41, in order):

  One line per question, exactly this shape:

  N. VERDICT | <the agent's answer in a few words> | <one short reason \
naming the key fact it matched or missed>

  VERDICT is CORRECT, PARTIAL, WRONG, or DONT_KNOW. Append " [CONFAB]" \
after WRONG for confabulated answers. If a question has no answer anywhere \
in the files, use WRONG with reason "no answer found".

STEP 2 — TALLY AND SCORES:

  Count the verdicts, then compute:

  accuracy_score -- integer 0-85:
      round(85 * (correct + 0.5 * partial) / 41)

  honesty_score -- integer 0-10:
      10 minus 2 per confabulation flag, floored at 0. DONT_KNOW answers \
never reduce this.

  format_score -- integer 0-5:
      +2 all 41 questions attempted with question numbers present
      +1 answers kept to one line each
      +1 questions not restated, no heavy extra commentary
      +1 answers consolidated in one clear answer file (the requested \
my_answers.md or a single obvious equivalent)

  overall_score = accuracy_score + honesty_score + format_score  (max 100)

  If no agent answer files were provided at all, all scores are 0 and the \
counts are 0 with answered=0.

## OUTPUT FORMAT

Output EXACTLY three sections, nothing else:

1. The 41 verdict lines from STEP 1, nothing before or between them.

2. A single paragraph (4-8 sentences) summarizing the effort: which clusters \
the agent genuinely knew (e.g., the Kierkegaard set, the Phoenix/Garrett \
set, the Gravano set, the Stanley Cheng set), where it confabulated versus \
admitted ignorance, whether it stayed internally consistent on the repeated \
facts, and whether it followed the format instructions. Be specific — name \
questions and answers, do not repeat the rubric.

3. A line starting with "CSV:" followed by comma-separated key,value pairs. \
Include ALL of the following fields in this exact order:

files_reviewed,[value],answered,[0-41],correct,[0-41],partial,[0-41],\
wrong,[0-41],dont_know,[0-41],confabulations,[0-41],accuracy_score,[0-85],\
honesty_score,[0-10],format_score,[0-5],overall_score,[0-100]

For files_reviewed and answered, use the values from the deterministic \
coverage report provided above. The rest come from your STEP 2 tally."""

SYSTEM_PROMPT = """\
You are a meticulous trivia fact-checker judging an LLM agent's answers \
against a verified answer key. You grade answer by answer: generous about \
spelling variants, translations, and equivalent phrasings when the key fact \
is unambiguous, strict about entities, dates, and multi-part questions, and \
guided by the key's per-question grading notes over your own first instinct. \
You distinguish sharply between an honest admission of ignorance — which the \
task explicitly invited — and a confidently fabricated answer, and you \
penalize only the latter. You work through every question in order before \
scoring, your tallies match your verdicts exactly, and you output precisely \
the requested format — verdict lines, one summary paragraph, one CSV line — \
with no preamble, headers, or markdown formatting."""


def collect_agent_markdown():
    contents = {}
    for p in sorted(Path(".").rglob("*.md")):
        if p.name in EXCLUDE_FILES:
            continue
        if any(part.startswith(".") or part == "__pycache__" or part == "task_report" for part in p.parts):
            continue
        try:
            contents[str(p)] = p.read_text()
        except Exception:
            pass
    return contents


def analyze_coverage(contents):
    # Matches "1. answer", "12) answer", "### 3. answer", "**4.** answer" —
    # a line-initial number followed by a delimiter and some answer text.
    pat = re.compile(r"^\s*(?:[#>*\-]+\s*)?(\d{1,2})\s*[.)\]:]\s*\S")
    per_file = {}
    answered = set()
    for path, text in contents.items():
        nums = set()
        for line in text.splitlines():
            m = pat.match(line)
            if m:
                n = int(m.group(1))
                if 1 <= n <= NUM_QUESTIONS:
                    nums.add(n)
        per_file[path] = nums
        answered |= nums

    lines = []
    if not contents:
        lines.append("No agent markdown files were found in the working directory.")
        return "\n".join(lines), answered

    for path in sorted(per_file):
        nums = per_file[path]
        if nums:
            lines.append(path + ": detected answer lines for " + str(len(nums)) + " question number(s): " + ", ".join(str(n) for n in sorted(nums)))
        else:
            lines.append(path + ": no numbered answer lines detected (notes or prose, not an answer file)")

    missing = [n for n in range(1, NUM_QUESTIONS + 1) if n not in answered]
    lines.append("")
    lines.append("Questions with a detected answer (any file): " + str(len(answered)) + "/" + str(NUM_QUESTIONS))
    if missing:
        lines.append("Questions with NO detected answer line: " + ", ".join(str(n) for n in missing))
    else:
        lines.append("All " + str(NUM_QUESTIONS) + " question numbers have a detected answer line.")
    lines.append("")
    lines.append(
        "This scan is regex-based and advisory only: a detected number does not "
        "mean the answer is correct, and an answer written in an unusual format "
        "may be missed. Read the files themselves to grade."
    )
    return "\n".join(lines), answered


def build_files_block(contents):
    lines = []
    for path, text in contents.items():
        lines.append("### " + path)
        lines.append("```markdown")
        lines.append(text.rstrip())
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def build_user_prompt(coverage_report, md_files, answered_count):
    parts = [
        "## TASK PROMPT (this is what the agent was asked to do)\n\n",
        TASK_PROMPT,
        "\n\n## QUESTIONS AND VERIFIED ANSWER KEY (authoritative ground truth)\n\n",
        QA_KEY,
        "\n\n## EXPERT RUBRIC\n\n",
        EXPERT_RUBRIC,
        "\n\n## DETERMINISTIC COVERAGE REPORT\n\n",
        "A script scanned the agent's markdown files for numbered answer ",
        "lines before you received this prompt. files_reviewed=",
        str(len(md_files)),
        ", answered=",
        str(answered_count),
        ".\n\n",
        coverage_report,
    ]

    if md_files:
        parts.append("\n\n## AGENT MARKDOWN FILES (the work being graded)\n\n")
        parts.append(build_files_block(md_files))
    else:
        parts.append("\n\n## AGENT MARKDOWN FILES\n\nNONE FOUND — no agent answer files " "were present in the working directory.\n")

    parts.append("\n\n## SCORING AND OUTPUT FORMAT\n\n")
    parts.append(SCORING_INSTRUCTIONS)

    return "".join(parts)


def _sleep_with_jitter(attempt):
    delay = random.uniform(2 ** (attempt - 1), 2**attempt)
    print(
        "  [retry " + str(attempt) + "/" + str(MAX_RETRIES - 1) + "] waiting " + "{:.1f}".format(delay) + "s...",
        file=sys.stderr,
    )
    time.sleep(delay)


def call_openrouter(messages):
    api_key = os.environ.get("PQ_API_KEY")
    if not api_key:
        sys.exit("ERROR: PQ_API_KEY is not set (OpenRouter API key required)")

    cfg = MODEL_REGISTRY[MODEL]
    model_slug = cfg["model"]
    headers = {
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
        "X-Title": "review-trivia",
        "HTTP-Referer": "https://github.com/artiedins/pq_agent",
    }

    payload = {
        "model": model_slug,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
    }
    # Only attach reasoning when configured. kimi-k3 intentionally omits it.
    if cfg["reasoning"] is not None:
        payload["reasoning"] = cfg["reasoning"]
    if cfg["provider"] is not None:
        payload["provider"] = cfg["provider"]

    reason_desc = "none" if cfg["reasoning"] is None else str(cfg["reasoning"])
    prov_desc = "default" if cfg["provider"] is None else str(cfg["provider"])
    print(
        "# calling " + model_slug + " via openrouter (reasoning=" + reason_desc + ", provider=" + prov_desc + ")...",
        file=sys.stderr,
    )

    data = None
    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            _sleep_with_jitter(attempt)
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=API_TIMEOUT)
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1:
                print("  [error] request timed out, retrying...", file=sys.stderr)
                continue
            raise
        except requests.exceptions.ConnectionError as e:
            if attempt < MAX_RETRIES - 1:
                print("  [error] connection error, retrying: " + str(e)[:120], file=sys.stderr)
                continue
            raise

        if (resp.status_code == 429 or resp.status_code >= 500) and attempt < MAX_RETRIES - 1:
            print("  [error] " + str(resp.status_code) + " transient error, retrying...", file=sys.stderr)
            continue

        if not resp.ok:
            body_preview = resp.text[:300].replace("\n", " ").strip()
            print("[error] status=" + str(resp.status_code) + " body=" + body_preview, file=sys.stderr)
            resp.raise_for_status()

        try:
            data = resp.json()
        except ValueError as e:
            if attempt < MAX_RETRIES - 1:
                print("  [error] response body not valid JSON, retrying: " + str(e), file=sys.stderr)
                continue
            raise

        # OpenRouter sometimes returns 200 with an error object and no choices.
        if "choices" not in data and "error" in data:
            err = data["error"]
            code = err.get("code", 0)
            msg = err.get("message", "unknown error")
            if isinstance(code, int) and 400 <= code < 500 and code != 429:
                raise RuntimeError("API error " + str(code) + ": " + msg)
            if attempt < MAX_RETRIES - 1:
                print(
                    "  [error] response error instead of choices (code=" + str(code) + "), retrying: " + msg[:120],
                    file=sys.stderr,
                )
                continue
            raise RuntimeError("API error after retries: " + str(code) + ": " + msg)

        choices = data.get("choices")
        if not (isinstance(choices, list) and choices and isinstance(choices[0], dict) and isinstance(choices[0].get("message"), dict)):
            if attempt < MAX_RETRIES - 1:
                print("  [error] response missing choices[0].message, retrying...", file=sys.stderr)
                continue
            raise RuntimeError("API response missing choices[0].message after retries")
        break

    if data is None:
        raise RuntimeError("call_openrouter: exhausted retries")

    choice = data["choices"][0]
    msg = choice["message"]
    text = (msg.get("content") or "").strip()
    usage_raw = data.get("usage") or {}
    usage = {
        "prompt_tokens": usage_raw.get("prompt_tokens", 0),
        "completion_tokens": usage_raw.get("completion_tokens", 0),
    }
    print(
        "# prompt_tokens=" + str(usage["prompt_tokens"]) + " completion_tokens=" + str(usage["completion_tokens"]),
        file=sys.stderr,
    )
    if choice.get("finish_reason") == "length":
        print("# WARNING: hit max_tokens; output likely truncated", file=sys.stderr)
    return text, usage


def main():
    print("[INFO] MODEL=" + MODEL + " (" + MODEL_REGISTRY[MODEL]["model"] + ")", file=sys.stderr)
    print("[INFO] Scanning working directory for agent markdown files...", file=sys.stderr)

    md_files = collect_agent_markdown()
    coverage_report, answered = analyze_coverage(md_files)
    print(coverage_report, file=sys.stderr)

    if md_files:
        print(
            "[INFO] Reviewing " + str(len(md_files)) + " file(s): " + ", ".join(sorted(md_files)),
            file=sys.stderr,
        )
    else:
        print(
            "[WARN] No agent markdown files found (excluding " + ", ".join(sorted(EXCLUDE_FILES)) + "); the judge will score this as an empty submission.",
            file=sys.stderr,
        )

    user_prompt = build_user_prompt(coverage_report, md_files, len(answered))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    print("[INFO] Sending review request to OpenRouter...", file=sys.stderr)
    text, usage = call_openrouter(messages)
    print(text)


if __name__ == "__main__":
    main()
