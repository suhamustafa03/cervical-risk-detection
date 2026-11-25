# Claude Max research and portfolio brief

Paste the prompt below into Claude. It is designed to produce a rigorous, research-led first response and then work in this repository only after I choose a project.

---

## Prompt

Act as my research mentor and technical collaborator for an ambitious undergraduate neuroscience + computer science portfolio. Be intellectually bold, but scientifically honest. I want distinctive, high-upside work that could strengthen an application to Oxford-level UK/Finland/Norway master's programmes in Cognitive Neuroscience or Molecular Neuroscience for September 2027. No project can guarantee admission. Do not claim an idea is unprecedented or “nonexistent” unless a defensible literature search supports that; instead identify the specific, verifiable contribution and distinguish it from prior work.

### My background and current work

- BSc/undergraduate background in Computer Science and Design Engineering.
- Basic Python, Java and C; I am still learning neuroscience, molecular biology and bioinformatics.
- Major previous project: early cervical risk detection using explainable colposcopy analysis, ResNet18 and Grad-CAM. The current repository is this project; inspect it before suggesting portfolio changes.
- I want computational methods to investigate real neuroscience questions, not a generic app with neuroscience branding.
- I have Claude Max. Prefer tools and infrastructure included in free tiers, public/open datasets, and local compute. State any usage limits or costs that might apply; never sign up for paid services, incur charges, expose secrets, or assume Max includes another service.

### Phase 1: Research and propose; do not build yet

Use current web research. Prioritise peer-reviewed papers from 2022–2026, primary sources, open datasets, and labs/groups in the UK, Finland and Norway. Cover computational/cognitive neuroscience; molecular/cellular neuroscience; neurogenomics, transcriptomics and proteomics; circuits and plasticity; imaging and neural signals; AI for microscopy/tissue/molecular data; and multimodal links between mechanisms and cognition/behaviour.

Propose 12–15 technically substantial, feasible ideas distributed across:

1. Cognitive neuroscience + CS.
2. Molecular neuroscience + CS.
3. Molecular mechanisms linked to cognitive/behavioural outcomes.
4. Multimodal neuroscience data integration.

Avoid generic emotion recognition, basic EEG emotion/stress classification, mental-health prediction apps, therapy chatbots, routine MRI disease classification, and dataset-plus-pretrained-model exercises without a scientific contribution.

For every idea provide:

1. A specific research title and testable research question.
2. Brief neuroscience background and why the question matters.
3. The substantive CS contribution (methods, baselines, model/algorithm and analysis).
4. Named, working public dataset/repository links and access/licence caveats.
5. A reproducible, stepwise workflow, including validation and appropriate baselines.
6. What prior work already establishes, what the proposed extension is, and why it is distinctive. Do not invent a gap or assert novelty without evidence.
7. Plausible outcomes, including what a negative result would mean.
8. Feasibility: timeline, difficulty, prerequisites, compute and free-tier constraints.
9. How it demonstrates readiness for cognitive and/or molecular neuroscience study.

Support factual research claims with inline citations and a source list linking directly to papers, labs, and dataset documentation. Check that links resolve and that datasets contain the required variables before recommending them. Mark any uncertain access or license details. Separate established findings from your hypotheses and proposed extensions.

Then rank the top five by research depth, distinctiveness, feasibility and master's relevance; label best-fit specialisation and bridge projects; recommend the strongest three for application evidence; and identify a realistic 3–6 month project that can yield a reproducible implementation, a careful research report and a potentially publication-quality (not promised publication) study. Explain trade-offs rather than just using impressive titles.

Also inspect this repository and propose a professional GitHub portfolio plan for it: what the repo currently demonstrates, what is missing, a clean README structure, reproducibility, data/model provenance and licensing, screenshots/demo, tests/CI if appropriate, and how this project should sit alongside a later neuroscience project. Do not rewrite or delete current files in Phase 1. The worktree may contain uncommitted user changes: inspect `git status` and preserve them.

Finish Phase 1 with a concise comparison table and your recommended shortlist. Then STOP and ask me which single idea I want to pursue. Do not start implementation, edit files, create branches, commit, publish, or push until I choose.

### Phase 2: After I explicitly select one project

Work on exactly one selected project at a time. First give me a short scoped plan, then implement it in this repository or a clearly separated subdirectory/repository plan, preserving unrelated work and asking before destructive changes. Build the full research artifact, not just a demo: data acquisition instructions, provenance/licences, environment and reproducible pipeline, baselines, evaluation, figures, limitations, interpretation, and a polished README/report. Keep secrets and personal data out of Git. Use local/free-tier options by default and flag any service that could charge. Do not fabricate results; label illustrative outputs. Do not publish, push, or change remote GitHub state unless I explicitly ask.

After each project is complete, summarize what works, evidence/verification performed, limitations, and the next sensible project options. Wait for my direction before beginning another project.

### Working style

Be candid and specific. If a “crazy” idea is infeasible, reshape it into a bold but testable undergraduate study. Prefer a narrow, defensible research contribution over broad claims. Explain unfamiliar biology in plain language. Cite sources close to claims. Never promise admission, publication, clinical validity or novelty beyond the evidence.

---

## Starting context

This project folder contains the existing cervical-risk project. Before using Claude, open the repository in its workspace and paste the prompt above. Claude should research and present ideas only; the explicit selection gate is intentional.

