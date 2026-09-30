"""Repeatable descriptive analysis of independent synthetic household samples (v3)."""
import argparse
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from schema import YEARS, LABELS, SCORES, HEAT_COLUMN, COMPONENTS, input_files, required_columns
from plots import heatmap, trends

ROOT = Path(__file__).resolve().parent

# ---- Настройки и кодировки (сверьте со справочником кодов обследования) ----
MIN_PAIRS = 30            # порог для отчётов (--min-pairs); меньше полных пар -> корреляция не считается
FIRST_AGE_YEAR = 2022     # с этого года в roster есть год рождения
CODES = {
    'U14': {1: 0, 2: 1},                 # 1 -> водопровод есть, 2 -> нет
    'GR31_2': {1: 1, 2: 1, 3: 0},        # 1,2 -> трудности есть, 3 -> нет; прочее (в т.ч. «не актуально») -> NaN
    'HEAT': {1: 0, 2: 1},                # 1 -> могут платить, 2 -> не могут
}
SETTLEMENT_LABELS = {1: 'Город', 2: 'Село'}
BINARY = ('utility_difficulty', 'heat_difficulty', 'no_indoor_water')
# Ошибки целостности, при которых расчёт останавливается
FATAL_SUFFIXES = ('_TE_mismatch', '_K_mismatch', '_unmatched_people', '_unmatched_households')
SELECTED = [('persons', 'area'), ('life_satisfaction', 'finance_satisfaction'),
            ('area_per_person', 'living_satisfaction'), ('utility_difficulty', 'heat_difficulty')]
SELECTED_SET = {frozenset(p) for p in SELECTED}


# ------------------------------- утилиты -------------------------------
def numeric(series):
    return pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan)


def recode(series, mapping):
    """Unspecified codes remain missing, rather than becoming zero."""
    return numeric(series).map(mapping).astype(float)


def clean_text(frame):
    """Strip whitespace in all text columns so keys and codes match across files."""
    return frame.apply(lambda c: c.str.strip() if pd.api.types.is_string_dtype(c) or c.dtype == object else c)


def read_csv(path):
    return clean_text(pd.read_csv(path, dtype=str))


def require_unique(frame, keys, name):
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f'{name}: missing or duplicate key {keys}')


def differs(a, b):
    """True where values differ; two missing values count as equal."""
    return (a != b) & ~(a.isna() & b.isna())


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def fmt(x, spec):
    return 'нет данных' if pd.isna(x) else format(x, spec)


def json_default(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    raise TypeError(f'Not serializable: {type(obj)}')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def spearman_pairs(frame, columns, min_pairs=3):
    """Spearman rho on each pair's own complete cases (average ranks for ties)."""
    records = []
    for a, b in combinations(columns, 2):
        pair = frame[[a, b]].dropna()
        rho = np.nan
        if len(pair) >= min_pairs and pair[a].nunique() > 1 and pair[b].nunique() > 1:
            ranks = pair.rank(method='average')
            rho = float(ranks[a].corr(ranks[b], method='pearson'))
        records.append({'variable_a': a, 'variable_b': b, 'rho': rho,
                        'n_pairs': len(pair),
                        'shared_formula': bool(COMPONENTS.get(a, {a}) & COMPONENTS.get(b, {b}))})
    return pd.DataFrame(records, columns=['variable_a', 'variable_b', 'rho', 'n_pairs', 'shared_formula'])


# ------------------------------ подготовка ------------------------------
def prepare(raw, year):
    """Validate a single year, then build one row per household with left joins."""
    for name, columns in required_columns(year).items():
        missing = set(columns) - set(raw[name].columns)
        if missing:
            raise ValueError(f'{year}/{name}: missing columns {sorted(missing)}')
    r, d, s, o = (raw[k].copy() for k in ('roster', 'dwelling', 'subject', 'assessment'))

    for name, frame in [('roster', r), ('subject', s), ('assessment', o)]:
        frame['NOMP'] = numeric(frame['NOMP'])
        nomp = frame.NOMP.dropna()
        if (nomp.le(0) | nomp.mod(1).ne(0)).any():
            raise ValueError(f'{year}/{name}: invalid person number')
        require_unique(frame, ['NOMER', 'NOMP'], name)
    for name, frame in [('dwelling', d), ('subject', s), ('assessment', o)]:
        require_unique(frame, ['NOMER'], name)
    for field in ['TE', 'K', 'KOL_CHL']:
        if (r.groupby('NOMER')[field].nunique(dropna=False) > 1).any():
            raise ValueError(f'{year}: inconsistent {field} within a household')

    # Возраст приближённый: год обследования минус год рождения.
    age_available = year >= FIRST_AGE_YEAR and 'GOD_ROJD' in r.columns
    r['age_approx'] = np.nan
    if age_available:
        birth = numeric(r.GOD_ROJD)
        r['age_approx'] = (year - birth).where(birth.between(year - 120, year) & birth.mod(1).eq(0))
    r['child'] = r.age_approx.lt(18).astype(float).where(r.age_approx.notna())

    hh = r.groupby('NOMER').agg(
        persons=('NOMP', 'size'), region=('TE', 'first'), settlement=('K', 'first'),
        reported_size=('KOL_CHL', 'first'), valid_ages=('age_approx', 'count'),
        children=('child', 'sum')).reset_index()
    hh['children'] = hh.children.where(hh.valid_ages.eq(hh.persons))
    hh['child_share'] = hh.children / hh.persons
    hh['family_group'] = np.select(
        [hh.child_share.isna(), hh.children.eq(0), hh.children.eq(1), hh.children.eq(2)],
        ['Возраст неизвестен', 'Без детей', '1 ребёнок', '2 ребёнка'], default='3+ детей')
    hh.insert(0, 'year', year)

    heat_col = HEAT_COLUMN[year]
    audit = {'year': year, 'source_rows': {k: len(v) for k, v in raw.items()},
             'households': len(hh), 'age_available': age_available,
             'heat_question_available': heat_col is not None,
             'reported_size_mismatch': int(differs(numeric(hh.reported_size), hh.persons.astype(float)).sum())}

    for name, frame in [('dwelling', d), ('subject', s), ('assessment', o)]:
        audit[name + '_unmatched_households'] = int((~frame.NOMER.isin(hh.NOMER)).sum())
        for source, target in [('TE', 'region'), ('K', 'settlement')]:
            check = frame[['NOMER', source]].merge(hh[['NOMER', target]], on='NOMER', validate='one_to_one')
            audit[f'{name}_{source}_mismatch'] = int(differs(check[source], check[target]).sum())
        if name != 'dwelling':
            joined = frame[['NOMER', 'NOMP']].merge(
                r[['NOMER', 'NOMP']], on=['NOMER', 'NOMP'], how='left', indicator=True, validate='one_to_one')
            audit[name + '_unmatched_people'] = int(joined['_merge'].ne('both').sum())

    failures = {k: v for k, v in audit.items() if k.endswith(FATAL_SUFFIXES) and v}
    if failures:
        raise ValueError(f'{year}: integrity checks failed: {failures}')
    if audit['reported_size_mismatch']:
        # Раньше эта проверка случайно считалась фатальной; теперь это предупреждение.
        print(f'{year}: предупреждение, KOL_CHL не совпадает с числом строк в roster у '
              f'{audit["reported_size_mismatch"]} семей', file=sys.stderr)

    # Семьи, где D002 (subject) и ocenka заполняли разные люди.
    respondents = s[['NOMER', 'NOMP']].merge(o[['NOMER', 'NOMP']], on='NOMER', suffixes=('_s', '_o'))
    mismatched = set(respondents.loc[respondents.NOMP_s.ne(respondents.NOMP_o), 'NOMER'])
    audit['d002_different_respondents'] = len(mismatched)

    for source, target in [('OB_PL', 'area'), ('J_PL', 'living_area'), ('KOL_K', 'rooms')]:
        value = numeric(d[source])
        valid = value.gt(0)
        if target == 'rooms':
            valid &= value.mod(1).eq(0)
        audit[source + '_excluded_nonmissing'] = int((d[source].notna() & ~valid).sum())
        d[target] = value.where(valid)
    audit['living_area_exceeds_total'] = int(d.living_area.gt(d.area).sum())
    d['no_indoor_water'] = recode(d.U14, CODES['U14'])
    audit['U14_codes'] = d.U14.fillna('<missing>').value_counts().to_dict()

    for source, target in SCORES.items():
        value = numeric(s[source])
        s[target] = value.where(value.isin(range(1, 11)))
        audit[source + '_excluded_codes'] = s.loc[
            s[source].notna() & ~value.isin(range(1, 11)), source].value_counts().to_dict()

    o['utility_difficulty'] = recode(o.GR31_2, CODES['GR31_2'])
    o['heat_difficulty'] = recode(o[heat_col], CODES['HEAT']) if heat_col else np.nan
    audit['GR31_2_codes'] = o.GR31_2.fillna('<missing>').value_counts().to_dict()
    if heat_col:
        audit['heat_raw_codes'] = o[heat_col].fillna('<missing>').value_counts().to_dict()
    o.loc[o.NOMER.isin(mismatched), ['utility_difficulty', 'heat_difficulty']] = np.nan

    for frame, cols, indicator in [
            (d, ['area', 'living_area', 'rooms', 'no_indoor_water'], 'has_dwelling'),
            (s, list(SCORES.values()), 'has_subject'),
            (o, ['utility_difficulty', 'heat_difficulty'], 'has_assessment')]:
        hh = hh.merge(frame[['NOMER'] + cols].assign(**{indicator: True}),
                      on='NOMER', how='left', validate='one_to_one')
        hh[indicator] = hh[indicator].eq(True)

    hh['d002_respondent_mismatch'] = hh.NOMER.isin(mismatched)
    hh['area_per_person'] = hh.area / hh.persons
    hh['persons_per_room'] = hh.persons / hh.rooms
    # Код типа поселения читается как число, поэтому '1' и '1.0' дают один результат.
    hh['settlement_label'] = numeric(hh.settlement).map(SETTLEMENT_LABELS).fillna('Неизвестно')
    audit['missing_by_variable'] = hh[list(LABELS)].isna().sum().to_dict()
    return hh, audit


# ------------------------------- отчёты --------------------------------
def describe_group(frame, year, grouping='all', group='Все'):
    rows = []
    for variable in LABELS:
        v = frame[variable].dropna()
        lo = hi = np.nan
        if variable in BINARY and len(v):
            lo, hi = wilson(int(v.sum()), len(v))
        rows.append({'year': year, 'grouping': grouping, 'group': group,
                     'variable': variable, 'households': len(frame), 'n_valid': len(v),
                     'mean': v.mean(), 'median': v.median(), 'p25': v.quantile(.25),
                     'p75': v.quantile(.75), 'p95': v.quantile(.95),
                     'ci_low': lo, 'ci_high': hi})
    return pd.DataFrame(rows)


def write_summary(output, yearly, correlations, audits, min_pairs=MIN_PAIRS):
    years = [a['year'] for a in audits]
    lines = ['# Жилищная уязвимость: предварительный анализ', '',
             'Независимые синтетические выборки, без весов. Это сравнение распределений, не панель семей.', '',
             '| Год | Домохозяйства | D006 | Разные респонденты D002 |', '|---|---:|---:|---:|']
    for a in audits:
        lines.append(f'| {a["year"]} | {a["households"]} | {a["source_rows"]["dwelling"]} | '
                     f'{a["d002_different_respondents"]} |')
    lines += ['', 'У несовпадающих респондентов показатели ocenka исключены; сами семьи сохранены.', '',
              '| Год | Медиана площади на человека, м² | Трудности оплаты ЖКУ, % [95% ДИ Уилсона] | Валидных ответов ЖКУ |',
              '|---|---:|---:|---:|']
    for year in years:
        a = yearly[(yearly.year == year) & (yearly.variable == 'area_per_person')].iloc[0]
        u = yearly[(yearly.year == year) & (yearly.variable == 'utility_difficulty')].iloc[0]
        n = int(u.n_valid)
        share = (f'{fmt(100 * u["mean"], ".2f")} [{fmt(100 * u.ci_low, ".2f")}; {fmt(100 * u.ci_high, ".2f")}]')
        lines.append(f'| {year} | {fmt(a["median"], ".2f")} | {share} | {n} |')

    lines += ['', f'## Выбранные корреляции Спирмена (пары с числом наблюдений от {min_pairs})', '',
              '| Год | Пара | ρ | Полных пар |', '|---|---|---:|---:|']
    for row in correlations.itertuples():
        if frozenset((row.variable_a, row.variable_b)) in SELECTED_SET:
            lines.append(f'| {row.year} | {LABELS[row.variable_a]} ↔ {LABELS[row.variable_b]} | '
                         f'{fmt(row.rho, ".3f")} | {row.n_pairs} |')

    lines += ['', '## Ограничения', '']
    if 2021 in years:
        lines += ['- В 2021 году нет года рождения; возраст, число и доля детей не восстанавливаются.',
                  '- GR32 в 2021 означает замену мебели. Показатель отопления доступен только с 2022 года.']
    lines += ['- Коды 89 и неприменимые ответы исключаются, а не превращаются в нули.',
              '- Возраст приближённый (год обследования минус год рождения), поэтому граница 18 лет неточна.',
              '- Изменились набор региональных кодов и состав выборки. Межгодовые сравнения регионов требуют согласования границ.',
              '- Корреляции рассчитаны отдельно для каждого года, без причинных выводов, p-value и поправок на дизайн обследования.',
              '- Общие компоненты формул могут сами порождать корреляцию; см. shared_formula.',
              '- Синтез может ослаблять зависимости. Изменения не являются официальной оценкой динамики населения.',
              '- В quality.json сохранены проверки ключей, служебные коды и число случаев, когда жилая площадь больше общей. '
              'Такие строки не удалены; жилая площадь не входит в матрицы.']
    (output / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')


def dump_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=json_default), encoding='utf-8')


# -------------------------------- запуск --------------------------------
def run(data_dir, output_dir, years, min_pairs=MIN_PAIRS, plots=True):
    years = sorted(set(years))
    if not years:
        raise ValueError('Select at least one year')
    missing = [str(data_dir / f) for y in years for f in input_files(y).values()
               if not (data_dir / f).is_file()]
    if missing:
        raise FileNotFoundError('Missing input files:\n' + '\n'.join(missing))

    prepared = []
    for year in years:
        raw = {k: read_csv(data_dir / p) for k, p in input_files(year).items()}
        for name, frame in raw.items():
            field = 'GOD' if name == 'roster' else 'GODO'
            if field in frame.columns and numeric(frame[field]).dropna().ne(year).any():
                raise ValueError(f'{name}: records do not match requested year {year}')
        hh, audit = prepare(raw, year)
        prepared.append((year, hh, audit))

    output = output_dir / ('years_' + '_'.join(map(str, years)))
    output.mkdir(parents=True, exist_ok=True)
    all_corr, all_desc, all_groups, all_strata, audits, frames = [], [], [], [], [], []
    for year, hh, audit in prepared:
        folder = output / str(year)
        folder.mkdir(exist_ok=True)
        correlations = spearman_pairs(hh, list(LABELS), min_pairs).assign(year=year)
        yearly = describe_group(hh, year)
        groups = pd.concat([describe_group(g, year, key, str(label))
                            for key in ['settlement_label', 'family_group', 'region']
                            for label, g in hh.groupby(key)], ignore_index=True)
        strata = pd.concat([spearman_pairs(g, list(LABELS), min_pairs).assign(year=year, settlement=label)
                            for label, g in hh.groupby('settlement_label')], ignore_index=True)
        hh.to_csv(folder / 'households.csv', index=False)
        correlations.to_csv(folder / 'correlations.csv', index=False)
        yearly.to_csv(folder / 'descriptive.csv', index=False)
        groups.to_csv(folder / 'groups.csv', index=False)
        strata.to_csv(folder / 'correlations_by_settlement.csv', index=False)
        dump_json(folder / 'quality.json', audit)
        if plots:
            heatmap(hh, correlations, folder / 'correlations.svg', year)
        all_corr.append(correlations); all_desc.append(yearly)
        all_groups.append(groups); all_strata.append(strata)
        audits.append(audit); frames.append(hh)
        print(f'{year}: {len(hh):,} households; D002 respondent mismatches: {audit["d002_different_respondents"]}')

    combined = pd.concat(frames, ignore_index=True)
    require_unique(combined, ['year', 'NOMER'], 'combined')
    combined.to_csv(output / 'households_all_years.csv', index=False)
    yearly = pd.concat(all_desc, ignore_index=True)
    correlations = pd.concat(all_corr, ignore_index=True)
    yearly.to_csv(output / 'yearly_summary.csv', index=False)
    correlations.to_csv(output / 'correlations_by_year.csv', index=False)
    pd.concat(all_groups, ignore_index=True).to_csv(output / 'groups_by_year.csv', index=False)
    pd.concat(all_strata, ignore_index=True).to_csv(output / 'correlations_by_year_and_settlement.csv', index=False)

    dump_json(output / 'run_info.json', {
        'years': years, 'min_pairs': min_pairs, 'data_dir': str(data_dir.resolve()),
        'versions': {'pandas': pd.__version__, 'numpy': np.__version__},
        'inputs': {str(y): {k: {'path': p, 'sha256': sha256(data_dir / p)}
                            for k, p in input_files(y).items()} for y in years}})
    write_summary(output, yearly, correlations, audits, min_pairs)
    if plots:
        trends(yearly, output / 'year_comparison.svg')
    print(f'Results: {output.resolve()}')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--years', type=int, nargs='+', choices=YEARS, default=list(YEARS))
    parser.add_argument('--min-pairs', type=int, default=MIN_PAIRS,
                        help='минимум полных пар для корреляции')
    parser.add_argument('--no-plots', action='store_true', help='не рисовать SVG')
    args = parser.parse_args()
    try:
        run(args.data_dir, args.output_dir, args.years, args.min_pairs, not args.no_plots)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f'Ошибка: {error}\n')
