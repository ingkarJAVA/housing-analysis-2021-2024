"""Semantic checks for year mappings, missingness, joins and pipeline outputs."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import json

from analyze import (clean_text, describe_group, differs, json_default, prepare, recode,
                     require_unique, run, spearman_pairs, wilson)
from schema import input_files


def fixture(year):
    # Household a has two people; b has one. Dwelling b is intentionally absent.
    roster = pd.DataFrame({'NOMER': ['a', 'a', 'b'], 'NOMP': ['01', '02', '01'],
        'TE': ['11', '11', '15'], 'K': ['1', '1', '2'], 'KOL_CHL': ['2', '2', '1']})
    if year >= 2022:
        roster['GOD_ROJD'] = [str(year-35), str(year-10), str(year-45)]
    dwelling = pd.DataFrame({'NOMER': ['a'], 'TE': ['11'], 'K': ['1'],
        'OB_PL': ['60'], 'J_PL': ['40'], 'KOL_K': ['2'], 'U14': ['1']})
    subject = pd.DataFrame({'NOMER': ['a', 'b'], 'NOMP': ['1', '1'], 'TE': ['11', '15'],
        'K': ['1', '2'], 'GR1': ['8', '89'], 'GR2': ['7', '6'], 'GR4': ['5', '4']})
    assessment = pd.DataFrame({'NOMER': ['a', 'b'], 'NOMP': ['01', '01'],
        'TE': ['11', '15'], 'K': ['1', '2'], 'GR31_2': ['1', '4'], 'GR32': ['2', '1']})
    return dict(roster=roster, dwelling=dwelling, subject=subject, assessment=assessment)


class AnalysisTests(unittest.TestCase):
    def test_2021_does_not_treat_furniture_as_heating_or_missing_age_as_zero(self):
        hh, audit = prepare(fixture(2021), 2021)
        self.assertTrue(hh.heat_difficulty.isna().all())
        self.assertTrue(hh.children.isna().all())
        self.assertTrue(hh.child_share.isna().all())
        self.assertFalse(audit['heat_question_available'])

    def test_2022_mapping_and_left_join(self):
        hh, _ = prepare(fixture(2022), 2022)
        hh = hh.set_index('NOMER')
        self.assertEqual(hh.loc['a', 'heat_difficulty'], 1)
        self.assertEqual(hh.loc['a', 'child_share'], .5)
        self.assertEqual(hh.loc['a', 'area_per_person'], 30)
        self.assertEqual(len(hh), 2)
        self.assertTrue(pd.isna(hh.loc['b', 'area']))
        self.assertTrue(pd.isna(hh.loc['b', 'utility_difficulty']))
        self.assertTrue(pd.isna(hh.loc['b', 'life_satisfaction']))

    def test_mismatched_respondent_does_not_drop_household(self):
        raw = fixture(2023)
        raw['assessment'].loc[0, 'NOMP'] = '02'
        hh, audit = prepare(raw, 2023)
        self.assertEqual(audit['d002_different_respondents'], 1)
        self.assertTrue(pd.isna(hh.loc[hh.NOMER.eq('a'), 'utility_difficulty'].iloc[0]))
        self.assertEqual(hh.loc[hh.NOMER.eq('a'), 'area'].iloc[0], 60)

    def test_expected_birth_column_cannot_silently_disappear(self):
        raw = fixture(2022)
        raw['roster'] = raw['roster'].drop(columns='GOD_ROJD')
        with self.assertRaisesRegex(ValueError, 'GOD_ROJD'):
            prepare(raw, 2022)

    def test_unknown_codes_are_missing(self):
        got = recode(pd.Series(['1', '2', '3', '4', None, '89']), {1: 1, 2: 1, 3: 0})
        self.assertEqual(got.dropna().tolist(), [1., 1., 0.])
        self.assertEqual(got.isna().sum(), 3)

    def test_pairwise_missing_and_ties(self):
        data = pd.DataFrame({'a': [1, 2, 2, 4, 999], 'b': [9, 7, 7, 1, np.nan]})
        row = spearman_pairs(data, ['a', 'b']).iloc[0]
        self.assertEqual(row.n_pairs, 4)
        self.assertAlmostEqual(row.rho, -1)

    def test_constant_variable_has_no_correlation(self):
        row = spearman_pairs(pd.DataFrame({'a': [1, 1, 1], 'b': [2, 3, 4]}), ['a', 'b']).iloc[0]
        self.assertTrue(pd.isna(row.rho))

    def test_duplicate_join_keys_rejected(self):
        with self.assertRaises(ValueError):
            require_unique(pd.DataFrame({'NOMER': ['1', '1']}), ['NOMER'], 'example')

    def test_shared_denominator_flag(self):
        data = pd.DataFrame({'child_share': [.1, .2, .3], 'area_per_person': [40, 30, 20]})
        self.assertTrue(spearman_pairs(data, list(data)).iloc[0].shared_formula)

    def test_min_pairs_threshold(self):
        data = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [4, 3, 2, 1]})
        self.assertAlmostEqual(spearman_pairs(data, ['a', 'b'], min_pairs=3).iloc[0].rho, -1)
        self.assertTrue(pd.isna(spearman_pairs(data, ['a', 'b'], min_pairs=30).iloc[0].rho))

    def test_reported_size_mismatch_is_warning_not_error(self):
        raw = fixture(2022)
        raw['roster']['KOL_CHL'] = ['3', '3', '1']
        with contextlib.redirect_stderr(io.StringIO()):
            _, audit = prepare(raw, 2022)
        self.assertEqual(audit['reported_size_mismatch'], 1)

    def test_region_mismatch_is_fatal(self):
        raw = fixture(2022)
        raw['dwelling'].loc[0, 'TE'] = '99'
        with self.assertRaisesRegex(ValueError, 'dwelling_TE_mismatch'):
            prepare(raw, 2022)

    def test_missing_in_both_files_is_not_a_mismatch(self):
        raw = fixture(2022)
        raw['roster'].loc[[0, 1], 'TE'] = None
        raw['dwelling'].loc[0, 'TE'] = None
        raw['subject'].loc[0, 'TE'] = None
        raw['assessment'].loc[0, 'TE'] = None
        prepare(raw, 2022)
        self.assertEqual(differs(pd.Series([None, 'a', 'a']), pd.Series([None, 'a', 'b'])).tolist(),
                         [False, False, True])

    def test_float_like_settlement_code(self):
        raw = fixture(2022)
        for name in raw:
            raw[name]['K'] = raw[name]['K'].map({'1': '1.0', '2': '2.0'})
        hh, _ = prepare(raw, 2022)
        self.assertEqual(hh.set_index('NOMER').loc['a', 'settlement_label'], 'Город')

    def test_whitespace_is_stripped(self):
        got = clean_text(pd.DataFrame({'NOMER': [' a ', 'b'], 'x': ['1 ', None]}))
        self.assertEqual(got.NOMER.tolist(), ['a', 'b'])

    def test_wilson_interval(self):
        lo, hi = wilson(50, 100)
        self.assertAlmostEqual(lo, .4038, places=3)
        self.assertAlmostEqual(hi, .5962, places=3)
        self.assertTrue(all(map(pd.isna, wilson(0, 0))))
        lo, hi = wilson(0, 50)
        self.assertEqual(lo, 0)
        self.assertGreater(hi, 0)

    def test_binary_variables_get_confidence_interval(self):
        hh, _ = prepare(fixture(2022), 2022)
        rows = describe_group(hh, 2022).set_index('variable')
        self.assertTrue(pd.isna(rows.loc['area', 'ci_low']))
        row = rows.loc['no_indoor_water']  # одна семья с данными о жилье, значение 0
        self.assertEqual(row.n_valid, 1)
        self.assertLessEqual(row.ci_low, row['mean'])
        self.assertGreaterEqual(row.ci_high, row['mean'])

    def test_full_pipeline_separates_years_even_if_ids_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for year in [2021, 2022]:
                for name, path in input_files(year).items():
                    target = root / 'data' / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    fixture(year)[name].to_csv(target, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                result = run(root / 'data', root / 'results', [2021, 2022])
            hh = pd.read_csv(result / 'households_all_years.csv')
            self.assertEqual(len(hh), 4)
            self.assertFalse(hh.duplicated(['year', 'NOMER']).any())
            pairs = pd.read_csv(result / 'correlations_by_year.csv')
            self.assertEqual(set(pairs.year), {2021, 2022})
            self.assertTrue((result / '2021' / 'correlations.svg').is_file())
            self.assertTrue(hh.loc[hh.year.eq(2021), 'heat_difficulty'].isna().all())
            json.loads((result / '2021' / 'quality.json').read_text(encoding='utf-8'))
            json.loads((result / 'run_info.json').read_text(encoding='utf-8'))
            self.assertIn('В 2021 году нет года рождения', (result / 'summary.md').read_text(encoding='utf-8'))

    def test_summary_omits_2021_notes_and_no_plots_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, path in input_files(2022).items():
                target = root / 'data' / path
                target.parent.mkdir(parents=True, exist_ok=True)
                fixture(2022)[name].to_csv(target, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                result = run(root / 'data', root / 'results', [2022], plots=False)
            self.assertNotIn('В 2021 году', (result / 'summary.md').read_text(encoding='utf-8'))
            self.assertFalse((result / 'year_comparison.svg').exists())
            self.assertFalse((result / '2022' / 'correlations.svg').exists())

    def test_missing_files_fail_before_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                run(root / 'missing', root / 'results', [2024])
            self.assertFalse((root / 'results').exists())


if __name__ == '__main__':
    unittest.main()
