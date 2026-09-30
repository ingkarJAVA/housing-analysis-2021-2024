"""Verified mappings for the supplied synthetic release (2021–2024)."""

YEARS = (2021, 2022, 2023, 2024)
DWELLING_NAMES = {2021: 'rio.csv', 2022: 'rio2022.csv',
                  2023: 'rio2023.csv', 2024: 'rio_2024.csv'}


def input_files(year):
    if year not in YEARS:
        raise ValueError(f'Unsupported year: {year}; expected {YEARS}')
    return {
        'roster': f'd008/{year}/kontr_k.csv',
        'dwelling': f'd006/{year}/{DWELLING_NAMES[year]}',
        'subject': f'd002/{year}/subject.csv',
        'assessment': f'd002/{year}/ocenka.csv',
    }


LABELS = {
    'persons': 'Человек в семье',
    'child_share': 'Доля детей (возраст ≈)',
    'area': 'Общая площадь, м²',
    'area_per_person': 'Площадь на человека',
    'persons_per_room': 'Человек на комнату',
    'life_satisfaction': 'Удовлетворённость жизнью',
    'living_satisfaction': 'Удовлетворённость условиями',
    'finance_satisfaction': 'Удовлетворённость финансами',
    'utility_difficulty': 'Трудности оплаты ЖКУ',
    'heat_difficulty': 'Не могут оплачивать тепло',
    'no_indoor_water': 'Нет водопровода в доме',
}
SCORES = {'GR1': 'life_satisfaction', 'GR2': 'living_satisfaction',
          'GR4': 'finance_satisfaction'}
# Column GR32 exists in 2021, but means furniture replacement, NOT heating!
HEAT_COLUMN = {2021: None, 2022: 'GR32', 2023: 'GR32', 2024: 'GR32'}
COMPONENTS = {'persons': {'persons'}, 'child_share': {'children', 'persons'},
              'area': {'area'}, 'area_per_person': {'area', 'persons'},
              'persons_per_room': {'persons', 'rooms'}}


def required_columns(year):
    return {
        'roster': ['NOMER', 'NOMP', 'TE', 'K', 'KOL_CHL'] +
                  (['GOD_ROJD'] if year >= 2022 else []),
        'dwelling': ['NOMER', 'TE', 'K', 'OB_PL', 'J_PL', 'KOL_K', 'U14'],
        'subject': ['NOMER', 'NOMP', 'TE', 'K', *SCORES],
        'assessment': ['NOMER', 'NOMP', 'TE', 'K', 'GR31_2'] +
                      ([HEAT_COLUMN[year]] if HEAT_COLUMN[year] else []),
    }
