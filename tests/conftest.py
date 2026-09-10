"""Společné nastavení testů.

`setup_console()` tu není z rozmaru: pipeline během běhu vypisuje varování
s diakritikou a na konzoli s kódováním cp1252 (výchozí stav na Windows) na
nich spadne dřív, než se stihne cokoli ověřit. Test by pak selhal na
kódování výpisu, ne na tom, co skutečně testuje.
"""
from magrag.console import setup_console

setup_console()
