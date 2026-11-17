"""Unit tests for rule-based assessment and tier/evidence rules."""

import unittest
import numpy as np
import pandas as pd

from exophca.science.assessment import (
    determine_tier,
    determine_evidence_quality,
    assess_planets,
    DISCLAIMER,
)


class AssessmentTests(unittest.TestCase):

    def test_tier_rules_a_b_c_d_u(self):
        # Tier A: conservative HZ and rocky_likely
        self.assertEqual("A", determine_tier("conservative", "rocky_likely"))
        # Tier B: (conservative HZ and uncertain) OR (optimistic_only HZ and radius <= 2.5)
        self.assertEqual("B", determine_tier("conservative", "uncertain"))
        self.assertEqual("B", determine_tier("optimistic_only", "rocky_likely"))
        self.assertEqual("B", determine_tier("optimistic_only", "uncertain"))
        # Tier C: any HZ (conservative or optimistic_only) and not_rocky_likely
        self.assertEqual("C", determine_tier("conservative", "not_rocky_likely"))
        self.assertEqual("C", determine_tier("optimistic_only", "not_rocky_likely"))
        # Tier D: outside HZ (too_hot or too_cold)
        self.assertEqual("D", determine_tier("too_hot", "rocky_likely"))
        self.assertEqual("D", determine_tier("too_cold", "uncertain"))
        # Tier U: undetermined
        self.assertEqual("U", determine_tier("undetermined", "rocky_likely"))
        self.assertEqual("U", determine_tier("conservative", "undetermined"))

    def test_evidence_rule_modelled_never_high_or_medium(self):
        for prov in ("modelled", "derived_unclear"):
            ev = determine_evidence_quality(prov, True, True)
            self.assertEqual("low", ev)

    def test_measured_no_radius_error_is_at_most_medium(self):
        ev = determine_evidence_quality("measured", False, True)
        self.assertIn(ev, ("medium", "low"))
        ev_full = determine_evidence_quality("measured", True, True)
        self.assertEqual("high", ev_full)

    def test_edge_cases_flux_limit_teff_out_of_range_missing_radius(self):
        df = pd.DataFrame({
            'pl_name': ['P1', 'P2', 'P3'],
            'pl_rade': [1.0, 1.0, np.nan],
            'pl_insol': [1.0, 1.0, 1.0],
            'st_teff': [5780.0, 2000.0, 5780.0],
            'hz_class': ['conservative', 'undetermined', 'undetermined'],
            'radius_provenance': ['measured', 'measured', 'missing'],
            'pl_radeerr1': [0.1, 0.1, np.nan],
            'pl_radeerr2': [-0.1, -0.1, np.nan],
            'pl_insolerr1': [0.1, 0.1, np.nan],
            'pl_insolerr2': [-0.1, -0.1, np.nan],
        })
        assessed = assess_planets(df)
        self.assertEqual('A', assessed.loc[0, 'tier'])
        self.assertEqual('U', assessed.loc[1, 'tier'])
        self.assertEqual('U', assessed.loc[2, 'tier'])
        self.assertIn("not evidence of habitability or life", DISCLAIMER)


if __name__ == '__main__':
    unittest.main()
