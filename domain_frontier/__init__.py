"""Search-based mapping of the IR+PE+NOM maximal-domain frontier.

Companion package to domain_expansion_experiments. We do NOT hand-implement the
paper's phi^IP mechanism; instead we treat "does an IR+PE+NOM marginal mechanism
exist on domain D?" as an existence question and search for a witness mechanism
directly (ML / random / exact). See the project plan for the soundness asymmetry:

  - finding a witness  PROVES feasibility (D in D_NOM);
  - failing to find one does NOT prove infeasibility (need exact search / theory);
  - an empty unambiguous IR+PE set on some profile PROVES infeasibility (cheap).

Modules:
  feasibility.py  -- mechanism-free unambiguous IR+PE existence (D_IRPE ceiling)
  sp_test.py      -- unambiguous strategy-proofness tester (D_SP map)
  search_nom.py   -- ML witness search + full-enum FOSD-NOM verification (D_NOM)
  map_frontier.py -- driver assembling the D_SP subset D_NOM subset D_IRPE map
"""
