#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.


lang2code = {
    'Albanian': 'sq', 'Afrikaans': 'af', 'Armenian': 'hy', 'Arabic': 'ar', 'Basque': 'eu',
    'Bulgarian': 'bg', 'Catalan': 'ca', 'Croatian': 'hr', 'Czech': 'cz',
    'Danish': 'da', 'Dutch': 'nl', 'English': 'en', 'Estonian': 'et', 'Finnish': 'fi', 'French': 'fr', 'German': 'de',
    'Greek': 'el', 'Hebrew': 'he', 'Hindi': 'hi', 'Hungarian': 'hu', 'Indonesian': 'id', 'Italian': 'it',
    'Latin': 'la', 'Latvian': 'lv', 'Lithuanian': 'lt', 'Norwegian_Bokmal': 'no', 'Norwegian_Nynorsk': 'nn', 'Persian': 'fa', 'Polish': 'pl',
    'Portuguese': 'pt', 'Romanian': 'ro', 'Russian': 'ru', 'Serbian': 'sr', 'Slovak': 'sk', 'Slovenian': 'sl', 'Spanish': 'es',
    'Swedish': 'sv', 'Turkish': 'tr', 'Ukrainian': 'uk', 'Urdu': 'ur'}
code2lang = {v: k for k, v in lang2code.items()}

family2langs = {
    'Albanian': ['Albanian'],
    'Slavic': ['Belarusian', 'Bulgarian', 'Croatian', 'Czech', 'Polish', 'Russian', 'Serbian', 'Slovak', 'Slovenian', 'Ukrainian'],
    'Germanic': ['Afrikaans', 'English', 'German', 'Dutch', 'Swedish', 'Norwegian_Bokmal', 'Norwegian_Nynorsk', 'Danish'],
    'Semitic': ['Arabic', 'Hebrew'],
    'Romance': ['Catalan', 'French', 'Galician', 'Italian', 'Latin', 'Romanian', 'Spanish', 'Portuguese'],
    'Uralic': ['Finnish', 'Estonian', 'Hungarian'],
    'Korean': ['Korean'],
    'Sino-Tibetan': ['Chinese'],
    'Japanese': ['Japanese'],
    'Turkic': ['Turkish', 'Kazakh'],
    'Indonesian': ['Indonesian'],
    'Indic': ['Hindi', 'Urdu', 'Marathi'],
    'Baltic': ['Lithuanian', 'Latvian'],
    'Dravidian': ['Telugu', 'Tamil'],
    'Basque': ['Basque'],
    'Persian': ['Persian'],
    'Viet': ['Vietnamese'],
    'Greek': ['Greek'],
    'Celtic': ['Irish', 'Breton'],
    'Armenian': ['Armenian'],
    'Tagalog': ['Tagalog'],
    'Yoruba': ['Yoruba'],
}

lang2family = {}

for family, languages in family2langs.items():
    for language in languages:
        lang2family[language] = family

lang2large_family = {}

for family, languages in family2langs.items():
    for language in languages:
        if len(languages) > 1:
            lang2large_family[language] = family
        else:
            lang2large_family[language] = 'Isolate'
