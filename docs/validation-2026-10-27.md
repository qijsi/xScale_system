# Documentation merge validation — 2026-10-27

- Active entry: `xScale.tex`; 13 reachable TeX files including the entry point.
- All active inputs exist; 9 distinct citation keys resolve to bibliography entries; no duplicate bibliography keys.
- All active references resolve to 12 unique labels; braces and begin/end environments balance.
- Paper, system repository, and CrystalLLM workspace copies of the new design document are byte-identical.
- All 10 archived sections match their previous committed versions byte-for-byte.
- `git diff --check` passes for both repositories.
- Existing runtime code was not changed; no new GPU experiments or implementation tests were run. The October 25 test counts remain historical.
- PDF build was not validated: no LaTeX compiler is installed, and an attempted temporary Tectonic download timed out. Static checks do not establish successful LaTeX compilation or final OSDI layout compliance.

The active manuscript explicitly marks proposed mechanisms and the evaluation protocol. Overleaf import/synchronization is left to the user; select `xScale.tex` as the main file.
