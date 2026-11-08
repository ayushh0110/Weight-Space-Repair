"""Weight-space programming beyond one failing prompt: objectives, task profiles and paired evaluation.

* `objective` - the goal file (task data split into practice and held-out, grader, protocol, guardrails);
* `graders`   - exact / contains / regex / math / detector:<group> scoring of a reply;
* `profile`   - triage verdicts over a dataset: which failure groups, how often, and what each is
                (decoding, budget, weight-level candidate), from tests that ran;
* `evaluate`  - the judge: original and patched model on the same items and seeds, the difference with
                its standard error, the guardrails, and a log of every look at the held-out set.
"""
