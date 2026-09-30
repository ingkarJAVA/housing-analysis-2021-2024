"""Small dependency-free SVG charts. Values and sample sizes are also saved as CSV."""
import html
import math
import pandas as pd
from schema import LABELS


def text(x, y, value, size=14, anchor='start', color='#183044'):
    return (f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{color}">{html.escape(str(value))}</text>')


def start(width, height):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            f'<rect width="{width}" height="{height}" fill="white"/>',
            '<g font-family="Arial,sans-serif">']


def save(parts, path):
    path.write_text('\n'.join(parts + ['</g></svg>']), encoding='utf-8')


def heatmap(frame, pairs, output, year):
    # No fake diagonal 1.0 for all-missing or constant fields.
    keys = [k for k in LABELS if frame[k].nunique() > 1]
    cell, x0, y0 = 60, 305, 140
    bottom = y0 + len(keys) * cell
    parts = start(1050, bottom + 120)
    parts += [text(25, 35, f'Жилищные условия · {year} · корреляции Спирмена', 23),
              text(25, 65, 'Синтетическая выборка · без весов · попарное исключение пропусков')]
    if year == 2021:
        parts.append(text(25, 92, 'Нет данных о возрасте детей и сопоставимого вопроса об отоплении.'))
    lookup = {(v.variable_a, v.variable_b): v.rho for v in pairs.itertuples()}
    for i, key in enumerate(keys):
        parts.append(text(x0 - 12, y0 + i*cell + 35, f'{i+1}. {LABELS[key]}', 13, 'end'))
        parts.append(text(x0 + i*cell + 30, y0 - 15, i + 1, 15, 'middle'))
        for j, other in enumerate(keys):
            v = 1.0 if i == j else lookup.get((key, other), lookup.get((other, key), float('nan')))
            strength = 0 if pd.isna(v) else min(1., abs(v))
            base = (40, 108, 165) if pd.notna(v) and v >= 0 else (198, 67, 61)
            color = '#%02x%02x%02x' % tuple(int(248*(1-strength) + c*strength) for c in base)
            label = '—' if pd.isna(v) else f'{0.0 if abs(v)<.005 else v:.2f}'
            parts.append(f'<rect x="{x0+j*cell}" y="{y0+i*cell}" width="59" height="59" fill="{color}"/>')
            parts.append(text(x0+j*cell+30, y0+i*cell+35, label, 14, 'middle',
                              'white' if strength > .6 else '#183044'))
    parts += [text(25, bottom+38, 'Синий: положительная связь. Красный: отрицательная. Число полных пар — в correlations.csv.'),
              text(25, bottom+66, 'Связи показателей с общими компонентами частично обусловлены самой формулой.'),
              text(25, bottom+94, 'Корреляции не доказывают причинность и не являются официальной статистикой.')]
    save(parts, output)


def trends(summary, output):
    parts = start(1100, 560)
    parts += [text(30, 35, 'Сравнение независимых выборок по годам', 24),
              text(30, 66, 'Разные семьи каждый год · синтетические данные · без весов обследования')]
    panels = [('area_per_person', 'median', 1, 'Площадь на человека: медиана, м²'),
              ('utility_difficulty', 'mean', 100, 'Трудности оплаты ЖКУ, % валидных ответов')]
    for panel, (variable, statistic, multiplier, title) in enumerate(panels):
        x0, top, baseline, width = 75+panel*535, 150, 430, 410
        data = summary.loc[summary.variable.eq(variable)].sort_values('year')
        values = data[statistic] * multiplier
        # Верх шкалы учитывает и верхнюю границу ДИ, если она есть (для долей).
        upper = values
        if statistic == 'mean' and 'ci_high' in data:
            upper = pd.concat([values, data['ci_high'] * multiplier])
        maximum = max(1., upper.max()) if upper.notna().any() else 1.
        maximum = math.ceil(maximum * 1.2)
        parts.append(text(x0, 117, title, 16))
        for tick in range(5):
            value = tick*maximum/4
            y = baseline - value/maximum*(baseline-top)
            parts.append(f'<line x1="{x0}" x2="{x0+width}" y1="{y}" y2="{y}" stroke="#e4eaf0"/>')
            parts.append(text(x0-12, y+5, f'{value:.1f}', 12, 'end'))
        step = width / max(1, len(data))
        for i, row in enumerate(data.itertuples()):
            v = getattr(row, statistic) * multiplier
            x = x0 + step*(i+.5)
            parts.append(text(x, baseline+25, row.year, 14, 'middle'))
            parts.append(text(x, baseline+45, f'n={row.n_valid}', 11, 'middle'))
            if pd.isna(v):
                parts.append(text(x, baseline-15, 'нет данных', 12, 'middle'))
                continue
            height = v/maximum*(baseline-top)
            parts.append(f'<rect x="{x-step*.3}" y="{baseline-height}" width="{step*.6}" height="{height}" fill="#286ca5"/>')
            lo, hi = getattr(row, 'ci_low', float('nan')), getattr(row, 'ci_high', float('nan'))
            label_y = baseline - height - 12
            if statistic == 'mean' and pd.notna(lo) and pd.notna(hi):
                y_lo = baseline - lo*multiplier/maximum*(baseline-top)
                y_hi = baseline - hi*multiplier/maximum*(baseline-top)
                parts.append(f'<line x1="{x}" x2="{x}" y1="{y_lo}" y2="{y_hi}" stroke="#183044" stroke-width="2"/>')
                for yy in (y_lo, y_hi):
                    parts.append(f'<line x1="{x-6}" x2="{x+6}" y1="{yy}" y2="{yy}" stroke="#183044" stroke-width="2"/>')
                label_y = y_hi - 8
            parts.append(text(x, label_y, f'{v:.2f}', 15, 'middle'))
    parts += [text(30, 484, 'Усы на графике ЖКУ: 95% доверительный интервал Уилсона, без учёта дизайна обследования.'),
              text(30, 512, 'Изменение состава выборки и синтез могут влиять на различия. Это не изменения у одних и тех же семей.'),
              text(30, 540, 'ЖКУ: ответы «не актуально», пропуски и несовпадающие респонденты D002 исключены из знаменателя.')]
    save(parts, output)
