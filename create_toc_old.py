import fitz
import json
import re


pdf_file = "data/ziva-2014-6.pdf"
output_file = "data/ziva_toc.json"


doc = fitz.open(pdf_file)


# stránky PDF, kde je obsah
toc_pages = [2]


# --------------------------------------------------
# pomocné funkce
# --------------------------------------------------

def clean(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_footer(text):

    bad = [
        "© Nakladatelství Academia",
        "SSČ AV ČR",
        "Přetisk článků",
        "http://ziva.avcr.cz",
        "www.ziva.avcr.cz"
    ]

    for b in bad:
        if b in text:
            text = text.split(b)[0]

    return text.strip(" ,;")


def is_roman(text):

    return bool(
        re.fullmatch(
            r"[IVXLCDM]+",
            text.strip()
        )
    )


def is_page_number(text):

    text = text.strip()

    return (
        text.isdigit()
        or is_roman(text)
    )


def is_page_span(span):

    txt = span["text"].strip()

    return (
        is_page_number(txt)
        and span["font"].startswith("MeliorCE")
        and "Bold" in span["font"]
    )


# --------------------------------------------------
# načtení spanů obsahu
# --------------------------------------------------

items_raw = []

current = None


for page_index in toc_pages:

    page = doc[page_index]

    blocks = page.get_text("dict")["blocks"]


    for block in blocks:

        if "lines" not in block:
            continue


        for line in block["lines"]:

            for span in line["spans"]:

                txt = span["text"].strip()

                if not txt:
                    continue


                # nová položka obsahu
                if is_page_span(span):

                    if current:
                        items_raw.append(current)


                    current = {
                        "page": txt,
                        "spans": []
                    }


                elif current:

                    # vynechání footerů
                    if (
                        "ziva.avcr.cz" not in txt
                        and "© Nakladatelství Academia" not in txt
                        and "Přetisk článků" not in txt
                    ):

                        current["spans"].append(
                            {
                                "text": txt,
                                "color": span["color"],
                                "font": span["font"],
                                "size": span["size"]
                            }
                        )


if current:
    items_raw.append(current)



# --------------------------------------------------
# rozdělení titul / autor
# --------------------------------------------------

def split_title_author(spans):

    if not spans:
        return None, None


    # odstranění prázdných
    spans = [
        s for s in spans
        if s["text"].strip()
    ]


    if not spans:
        return None, None


    # první span bereme jako titulový
    title_color = spans[0]["color"]


    title = []
    author = []


    author_started = False


    for span in spans:

        txt = span["text"].strip()

        if not txt:
            continue


        if (
            not author_started
            and span["color"] == title_color
        ):

            title.append(txt)

        else:

            author_started = True
            author.append(txt)


    return (
        clean(" ".join(title)),
        clean(" ".join(author))
        if author else None
    )



# --------------------------------------------------
# vytvoření výsledného TOC
# --------------------------------------------------

toc = []


for item in items_raw:

    title, author = split_title_author(
        item["spans"]
    )


    if title:

        toc.append(
            {
                "printed_page": item["page"],
                "title": remove_footer(title),
                "author": remove_footer(author)
                if author else None
            }
        )


# --------------------------------------------------
# uložení
# --------------------------------------------------

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        toc,
        f,
        ensure_ascii=False,
        indent=2
    )


print(
    f"Uloženo {len(toc)} položek do {output_file}"
)


for x in toc:

    print(
        x["printed_page"],
        "|",
        x["title"],
        "|",
        x["author"]
    )
