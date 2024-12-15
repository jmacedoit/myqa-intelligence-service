
import re

locale_region_language_name = {
    "en": {
        "default": "english"
    },
    "pt": {
        "default": "português",
        "br": "português do brasil",
        "pt": "português de portugal"
    },
    "es": {
        "default": "español"
    },
    "fr": {
        "default": "français"
    },
    "de": {
        "default": "deutsch"
    },
    "it": {
        "default": "italiano"
    }
}

def get_language_name(locale_region: str) -> str:
    locale_region = locale_region.lower()

    elements = re.split(r"[_-]", locale_region);

    locale = elements[0]
    region = elements[1] if len(elements) > 1 else "default"

    return locale_region_language_name[locale][region]
