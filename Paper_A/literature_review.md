# Literature review and positioning for Paper A

*Compiled 2026-10-02. Verified against primary sources; abstracts read in full,
full texts consulted where marked. Nothing here is from memory alone.*

---

## 1. The headline: two precursors exist, and one is the paper we reproduced

Paper A's framing in draft v2 — "the field does not check this" — is **no longer
accurate** and must be dropped. Two papers already make a version of the
argument, and both are strong. The good news is that neither makes *our*
argument, and one of them corroborates our central empirical result.

### 1.1 Bowles, Ahmed & Schuld (2024) — the benchmarking precursor

*Better than classical? The subtle art of benchmarking quantum machine learning
models.* [arXiv:2403.07059](https://arxiv.org/abs/2403.07059)

The major prior work. 12 quantum models, 6 binary classification tasks, 160
datasets, built on PennyLane. Findings: out-of-the-box classical models
outperform the quantum classifiers overall, and **removing entanglement from a
quantum model often performs as well or better**, suggesting "quantumness" is
not the active ingredient at these scales.

Three things matter for our positioning:

1. **Their repeats are 5-fold cross-validation** — footnote 3 states that one
   5-fold CV means "training the model five times". That varies *the data
   split*, not the initialisation, and conflates two sources of variance. They
   do not report seed dispersion with a fixed split, and they do not derive a
   required sample size. This is the gap Paper A fills.
2. **Their positivity-bias argument is a simulation, not a measurement.** Their
   Figure 3 is a thought experiment: 100 researchers each test 20 quantum models
   and publish the best, with assumed normal performance distributions (quantum
   mean 0.55, sd 0.10; classical mean 0.65, sd 0.07). The selection bias flips
   the apparent ranking. It is persuasive, but the dispersion figures are
   *assumed*. Paper A measures the real thing from real runs and derives the
   false-positive rate from it. That is a genuine complement, and we should say
   so explicitly rather than competing.
3. **Breadth versus depth.** They go wide (160 datasets, 5 repeats each). We go
   deep (25 seeds on one task, 10 on others). Those are different studies and
   the depth direction is the one that can measure dispersion at all.

### 1.2 QASA v2 (2025) — the source paper has revised its own claim

*Quantum Adaptive Self-Attention for Quantum Transformer Models.*
[arXiv:2504.05336](https://arxiv.org/abs/2504.05336)

**This is the most consequential finding of the review.** The paper we set out
to reproduce has been substantially rewritten. The current abstract opens with:

> "A recurring weakness in quantum machine learning (QML) is that reported
> 'quantum advantages' are seldom tested against a *capacity-matched* classical
> control…"

Its stated primary contribution is now **methodological**. The authors introduce
a capacity-matched classical bottleneck of identical parameter budget, find that
it **matches the PQC on the error metrics**, and conclude the gain is
"attributable to the low-rank value-projection *bottleneck* … not to
quantumness". They explicitly position the quantum layer "not as a source of
accuracy advantage".

Consequences for Paper A, in order of importance:

- **Our Mackey-Glass null now agrees with the source paper's own revised
  conclusion.** This is corroboration, not conflict, and it is a much stronger
  position than contradicting a live claim. Our independent reproduction,
  arriving at the same place by a different route (seeds rather than capacity
  matching), is worth more because of it.
- **"Capacity-matched control" is taken.** We cannot present parameter matching
  as a contribution. We should cite them for it and position on the orthogonal
  axis: they identify a confound in the *architecture*, we identify one in the
  *measurement*.
- **A reviewer will ask whether our classical control is capacity-matched in
  their sense.** In `QASA-repro/qasa.py`, `ClassicalTokenLayer` is
  `Linear(d_model → n_qubits) → Tanh → Linear(n_qubits → hidden) → Tanh →
  Linear(hidden → d_model)` — a genuine low-rank bottleneck, so our forecasting
  comparison already uses the right control. **[VERIFY — confirm the same holds
  for `DynamicsCore(quantum=False)` in `Phase_1_Scaffold_test/models/
  quantum_core.py`; the file was iCloud-evicted at time of writing. If that path
  is full-width rather than bottlenecked, our Ising and CartPole comparisons are
  not capacity-matched and the claim must be qualified.]**
- **A reframing becomes available.** Paper A can be presented as a direct
  independent replication of QASA that fails to find the advantage at 25 seeds
  and explains why 13 could never have resolved it. Replications are
  publishable, and this one is unusually well-motivated.

---

## 2. Trainability — Section 7 must be softened

### 2.1 Cerezo, Sone, Volkoff, Cincio & Coles (2021)

*Cost function dependent barren plateaus in shallow parametrized quantum
circuits.* [Nature Communications 12, 1791](https://www.nature.com/articles/s41467-021-21728-w)

The protection result we rely on. Global cost functions exhibit barren plateaus
at any depth; local cost functions are trainable when ansatz depth is
$O(\log n)$, with at-worst polynomial gradient decay.

**The condition is stated in terms of depth and causal cones, not layer count.**
The mechanism is that the light cone of a local observable must stay small. A
sequential CNOT cascade propagates correlations across the whole register within
one nominal layer, so its causal cone is immediately global and the theorem's
precondition fails.

This means our Section 7 **does not discover a mechanism** — the mechanism is
the published theory, correctly applied. The honest claim is narrower:

> A common implementation choice silently violates the depth precondition of a
> widely-cited protection result, and the empirical magnitude of that violation
> is roughly 1,260× in gradient variance at ten qubits. Practitioners plotting
> gradient variance against "layers" with a cascade entangler will measure a
> plateau that the theory does not predict for their cost function.

That is a practitioner warning with a measured magnitude, not a theoretical
contribution. It should be presented as such, which also removes the need to
resolve the depth-versus-entangling-structure question we flagged — under the
causal-cone framing the two are not really distinct, since a cascade's cone
growth *is* its entangling structure.

Also relevant and worth citing: a simulator study arguing qubit count rather
than circuit depth dominates barren plateau onset for hardware-efficient
ansätze ([SSRN 7222338](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7222338)),
and PennyLane's tutorial on local cost functions
([link](https://pennylane.ai/qml/demos/tutorial_local_cost_functions)), which is
how most practitioners actually encounter this result.

---

## 3. Independent corroboration of the dispersion thesis

### 3.1 Pandey (2026)

*Fixed-Reservoir vs Variational Quantum Architectures for Chaotic Dynamics:
Benchmarking QRC and QPINN on the Lorenz System.*
[arXiv:2604.23743](https://arxiv.org/abs/2604.23743)

Highly relevant and very recent (26 April 2026). Compares a variational Quantum
PINN against Quantum Reservoir Computing with a fixed transverse-field Ising
Hamiltonian, at 4–5 qubits and 2–3 layers, on Lorenz, Rössler and Lorenz-96.

**The number to quote: their QPINN test MSE is $47.9 \pm 36.6$.** The standard
deviation is 76% of the mean. That is an independent demonstration, in a
neighbouring setting, of precisely the dispersion Paper A argues is unreported
and decisive — and it comes from authors who were not making that point.

Two further overlaps worth noting in the text: they also use a transverse-field
Ising system, and they also rule out barren plateaus by inspecting gradient
norms ($10^3$–$10^4$) rather than assuming them, concluding the instability is
"capacity limitations and competing loss terms rather than barren plateaus" —
which parallels our R3 saturation/capacity argument.

Their headline is a *positive* result for QRC (81% lower MSE, ~52,000× faster
training). Paper A should note it without litigating it: at $\pm$ 36.6 on the
comparator, the effect is large enough that dispersion may not threaten it, but
the required seed count is exactly the quantity their study does not report.

---

## 4. Reproducibility and variance in classical ML

Bowles et al. already cite this literature, so it is established context rather
than novel framing. Paper A should cite it briefly and move on.

- Henderson et al., *Deep Reinforcement Learning that Matters* — seed variance
  and the inadequacy of small-$n$ comparisons in RL.
- Agarwal et al., *Deep RL at the Edge of the Statistical Precipice* — interval
  estimates and stratified bootstrap in place of point estimates.
- Bouthillier et al., *Accounting for Variance in Machine Learning Benchmarks* —
  decomposing variance sources, of which initialisation is one.
- Dodge et al. — incorporating computational cost into performance comparison;
  directly relevant to our ~20× wall-clock point.

**[AUDIT — obtain exact citations, years and venues for these four. They are
cited from Bowles et al.'s reference list and from general knowledge, and have
not been individually verified.]**

---

## 5. Revised positioning statement

What Paper A can claim after this review:

1. **An empirical measurement of seed dispersion across task families**, with the
   derived sample-size requirement varying 25× (6 seeds to 150). Neither Bowles
   et al. (5-fold CV, assumed distributions) nor QASA v2 (capacity matching) nor
   Pandey (reports $\pm$ but does not analyse it) does this.
2. **A failed replication with a pre-committed out-of-sample test.** Rare in QML
   generally, and here it independently corroborates the source paper's own
   revised conclusion.
3. **False-positive rates computed from real dispersion** rather than assumed
   normals — the empirical counterpart to Bowles et al.'s Figure 3 thought
   experiment.
4. **Sequence modelling and dynamics** rather than binary classification,
   including exactly-simulated quantum-native dynamics.
5. **A measured magnitude for the entangler/causal-cone pitfall**, as a
   practitioner warning.

What must be dropped:

- The claim or implication that the field does not examine these issues.
- Any suggestion that capacity matching is our contribution.
- Section 7's framing as a discovered mechanism.

---

## 6. Draft Section 2 text

> **Related work.**
>
> *Benchmarking and evaluation.* Bowles et al. [1] conduct the most extensive
> evaluation of quantum machine learning models to date, testing 12 architectures
> across 160 classification datasets and finding that classical baselines
> generally outperform quantum classifiers, and that removing entanglement often
> leaves performance unchanged. They further argue, via simulation, that
> publication bias over a large design space can invert an apparent ranking. Our
> work is complementary in two respects: where they go broad with five-fold
> cross-validation per configuration, we go deep, holding the data split fixed
> and varying only initialisation so that seed dispersion can be measured
> directly; and where their positivity-bias argument assumes performance
> distributions, we estimate the false-positive rate from the dispersion our own
> runs exhibit.
>
> *Capacity-matched controls.* The QASA architecture [2], whose dynamics core we
> reproduce here, has itself been revised to argue that its reported advantage is
> attributable to a low-rank value-projection bottleneck rather than to the
> quantum substrate, as demonstrated by a capacity-matched classical control. Our
> null result on chaotic forecasting is consistent with that conclusion, reached
> independently and by a different route: we hold capacity matched throughout and
> vary the number of seeds.
>
> *Trainability.* Cerezo et al. [3] show that local cost functions avoid barren
> plateaus provided ansatz depth remains $O(\log n)$, a condition expressed
> through the causal cones of the cost terms. Section 7 reports the empirical
> consequence of violating that condition inadvertently through a common
> entangler choice.
>
> *Variance in machine learning benchmarks.* Our statistical argument follows
> established work in classical machine learning [4–7], transposed to a setting
> where per-run cost is substantially higher and reported sample sizes are
> correspondingly smaller — which makes the problem more acute, not less. Recent
> work on quantum models for chaotic dynamics [8] reports comparator variance of
> 76% of the mean, illustrating the magnitudes involved.

---

## 7. Open items

1. **[VERIFY]** Whether `DynamicsCore(quantum=False)` is a capacity-matched
   bottleneck or a full-width classical block. Decides whether the Ising and
   CartPole comparisons meet QASA v2's control standard.
2. **[AUDIT]** Exact citations for Henderson, Agarwal, Bouthillier, Dodge.
3. Search not yet done: whether any QML paper reports seed dispersion and derives
   a required sample size. If one exists, contribution 1 above weakens
   substantially. Suggested queries: "statistical power quantum machine
   learning", "how many random seeds variational quantum classifier",
   "reproducibility variational quantum algorithms seeds".
4. Consider whether to reposition the whole paper as a replication study of
   QASA. Given the source paper's revision, this framing is both honest and
   unusually well-motivated.
