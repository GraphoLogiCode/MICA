# MICA poster — four text fixes (for Claude Design)

Task: apply four wording fixes to the existing MICA poster. **Text-only edits.**
Keep everything else exactly as it is: KaTeX serif look, same colors
(blue #2E6EB4, gray #8C8C8C, green #3C915A, red #B4463C), same layout, same
figures, same tables. Do not move, resize, or restyle any box.

---

## Fix 1 — Pipeline column, "embodied agent" box (tense contradiction)

The poster says the agent "places ONE block" but another column correctly says
it has never placed a block. Change the verb to capability, not fact.

OLD:
> follows, scans, voices offers · places ONE block only on your chat "yes"

NEW:
> follows, scans, voices offers · may place ONE block, only after your typed
> "yes" — route armed, never yet exercised

Reason: the agent has never placed a block (pinned fact). Present tense
"places" contradicts the What-failed bullet "The agent has never placed a
block. Zero, ever."

## Fix 2 — Finding 2 body text (overstated number)

OLD:
> label-free pretraining on real sessions closed a 1.25-nat domain gap
> (−37% prediction error)

NEW:
> label-free pretraining on real sessions cut real-session prediction error by
> 1.25 nats (−37%); the real-vs-scripted gap shrank from 1.65 to 0.45 nats

Reason: 1.25 nats is the drop in real-session error (3.42 → 2.17). The
domain gap (real minus scripted) went from 1.65 to 0.45 — narrowed, not
closed, and it was never 1.25. Verification numbers:
- real NLL: 3.4228 → 2.1708 (drop 1.25 nats = −37%)
- scripted NLL: 1.7683 → 1.7246
- gap before: 3.4228 − 1.7683 = 1.65; gap after: 2.1708 − 1.7246 = 0.45

## Fix 3 — Finding 1 headline (name the task)

OLD:
> Finding 1 — Intent inference is necessary off-template

NEW:
> Finding 1 — Intent inference is necessary for knowing WHICH GOAL
> off-template

(Style the emphasis however fits the heading — italics or small caps is fine;
the words "for knowing which goal" must appear.)

Reason: a reader will pit this headline against the What-failed bullet "The
belief is not prediction fuel … changes nothing." Both are true because they
are different tasks: belief is necessary for goal classification
(judgment), useless as forecasting input. The headline must name its task so
the two statements cannot be read as contradicting.

## Fix 4 — Finding 2 body text (one session was excluded)

OLD:
> contested and discarded sessions included, because no label enters the
> sample

NEW:
> contested and discarded sessions included (one session excluded for a
> corrupted capture), because no label enters the sample

Reason: one discarded session (210003) was quarantined for a truncated
capture and excluded from pretraining. So 8 of 9 discarded sessions were
included, and the one exclusion was for data integrity, not labels. The
poster's honesty standard requires saying so.

---

## Do NOT change

- The earliness row (0.990 → 0.897 shown green) stays as is — the metric
  improved without crossing the pre-registered bar, and the What-failed
  bullet "Early recognition: unproven" already covers that honestly.
- All other numbers, claims, figures, citations [1]–[3], and the METHOD strip
  are correct and stay untouched.
