"""Konzole: vynuť UTF-8 na stdout/stderr.

Proč to tu je: všechny hlášky pipeline jsou česky a na Windows má konzole
ve výchozím stavu kódování cp1252. První `print()` s diakritikou pak spadne
na `UnicodeEncodeError` - a to ještě předtím, než se stihne zpracovat jediná
stránka PDF:

    UnicodeEncodeError: 'charmap' codec can't encode character '\\u010d'

Ve Spyderu/IPythonu se to neprojeví (ty mají stdout v UTF-8), takže je to
přesně ten druh chyby, kterou autor nikdy nevidí a každý, kdo si projekt
naklonuje, do ní narazí do dvou sekund. Řeší se to jedním voláním na začátku
každého vstupního bodu, ne návodem v README, ať to nejde zapomenout.
"""
import sys


def setup_console() -> None:
    """Přepni stdout/stderr na UTF-8 s náhradou neznámých znaků.

    `errors="replace"` je tu záměrně: hlášky pipeline občas vypisují úryvky
    textu ze zdrojového PDF, kde se může objevit exotický znak (řecká
    písmena v biologických názvech, matematické symboly). Spadnout kvůli
    výpisu diagnostiky by bylo horší než vypsat otazník.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:  # None u přesměrovaného/obaleného streamu
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass  # stream nejde překonfigurovat - lepší běžet dál
