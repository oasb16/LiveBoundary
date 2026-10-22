# Provenance and related work

LiveBoundary is an original, compact implementation of ordinary gradient training plus explicit
finite-output constraint mass, evaluation gates, and checkpoint transactions. It borrows no source
code or claimed algorithm from the work below.

- [Pylon](https://pylon-lib.github.io/) compiles imperatively expressed output constraints into
  differentiable losses and integrates them with PyTorch. Its
  [repository](https://github.com/pylon-lib/pylon) is Apache-2.0. LiveBoundary instead uses a small
  closed JSON-like expression set and exact finite categorical enumeration. There is no affiliation.
- [Elastic Weight Consolidation](https://arxiv.org/abs/1612.00796) slows learning on parameters judged
  important to earlier tasks. LiveBoundary does not implement EWC; its initial preservation option is
  output distillation on designated stable probes plus optional replay.
- [MEND](https://arxiv.org/abs/2110.11309) learns auxiliary editor networks that transform gradients
  for fast targeted model edits. [AlphaEdit](https://arxiv.org/abs/2410.02355) projects editing
  perturbations into a preserved-knowledge null space. LiveBoundary does neither: it runs direct
  gradients on selected native model parameters and measures collateral effects afterward.
- [Test-Time Training layers](https://arxiv.org/abs/2407.04620) make a sequence model's hidden state
  itself a model updated by self-supervised learning. [Titans](https://arxiv.org/abs/2501.00663)
  introduces neural long-term-memory modules for sequence architectures. LiveBoundary instead makes
  persistent, explicitly checkpointed changes to the deployed task model after verified supervision.
- [SEAL](https://arxiv.org/abs/2506.10943) studies language models generating their own fine-tuning
  data and update directives, trained through downstream reward. LiveBoundary never lets unverified
  model generations authorize targets; callers must supply verified examples or explicit rules.

These comparisons locate the project. They do not support novelty, scale, or reliability claims.
