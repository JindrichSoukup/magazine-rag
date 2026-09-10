# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 13:14:42 2026

@author: Jindrich
"""

import fitz  # PyMuPDF
import json
from collections import Counter

pdf_file = "data/ziva-2014-6.pdf"
output_file = "data/ziva_blocks.json"

doc = fitz.open(pdf_file)


def classify_block(spans):
    """
    Určí hrubý typ bloku podle fontu a velikosti.
    """

    sizes = [round(s["size"], 1) for s in spans]
    fonts = [s["font"] for s in spans]

    max_size = max(sizes)
    main_font = Counter(fonts).most_common(1)[0][0]

    if max_size >= 20 and "Bold" in main_font:
        return "title"

    if 14 <= max_size < 20 and "Bold" in main_font:
        return "heading"

    if max_size <= 7.5:
        return "caption"

    if max_size >= 8 and max_size <= 10:
        return "body"

    return "other"


def clean_text(text):
    """
    Základní čištění.
    Zatím velmi opatrné.
    """

    text = text.replace("\u00ad", "")  # soft hyphen

    # odstranění zalomení slov typu:
    # variabi-
    # lita
    text = text.replace("-\n", "")

    # řádkové zlomy převést na mezery
    text = text.replace("\n", " ")

    # vícenásobné mezery
    text = " ".join(text.split())

    return text.strip()


def join_spans(spans):
    """
    Spojí span texty a doplní mezery tam,
    kde je PDF vynechalo.
    """

    result = ""

    for i, span in enumerate(spans):

        text = span["text"]

        if i > 0:
            prev = spans[i-1]["text"]

            # pokud předchozí text nekončí mezerou
            # a nový nezačíná interpunkcí
            if (
                not result.endswith((" ", "\n"))
                and not text.startswith((".", ",", ";", ":", ")", "]"))
                and not prev.endswith("-")
            ):
                result += " "

        result += text

    return result


blocks_out = []


for page_num, page in enumerate(doc, start=1):

    data = page.get_text("dict")

    for block_id, block in enumerate(data["blocks"]):

        # obrázky ignorujeme
        if "lines" not in block:
            continue

        spans = []

        for line in block["lines"]:
            for span in line["spans"]:
                if span["text"].strip():
                    spans.append(span)

        if not spans:
            continue

        raw_text = join_spans(spans)

        text = clean_text(raw_text)

        if not text:
            continue

        sizes = [
            round(s["size"], 1)
            for s in spans
        ]

        fonts = [
            s["font"]
            for s in spans
        ]

        record = {
            "page": page_num,
            "block_id": block_id,
            "type": classify_block(spans),
            "text": text,
            "font": Counter(fonts).most_common(1)[0][0],
            "font_size": max(sizes),
            "bbox": [
                round(x, 1)
                for x in block["bbox"]
            ]
        }

        blocks_out.append(record)


with open(output_file, "w", encoding="utf-8") as f:
    json.dump(
        blocks_out,
        f,
        ensure_ascii=False,
        indent=2
    )


print(
    f"Uloženo {len(blocks_out)} bloků do {output_file}"
)
