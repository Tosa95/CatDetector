from typing import Any, Iterable

from cat_detector.paths import load_uc_model_data
from cat_detector.templates.load_template import load_template


def chunk(data: Iterable[Any], n: int):
    curr = []
    for d in data:
        curr.append(d)
        if len(curr) >= n:
            yield curr
            curr = []
    yield curr


def prepare_model_for_uc(name, model_data, destination_file):
    model_data_len = len(model_data)

    model_data_stringed = ["0x%0.2X" % b for b in model_data]
    model_data_stringed = [", ".join(c) for c in chunk(model_data_stringed, 12)]

    model_data_stringed = ",\n".join(model_data_stringed)

    converted = load_template("uc_model_template").render(model_name=name, data=model_data_stringed,
                                                          data_len=model_data_len)
    with open(destination_file + ".cpp", "wt") as f:
        f.write(converted)

    header = load_template("uc_model_template_h").render(model_name=name)
    with open(destination_file + ".h", "wt") as f:
        f.write(header)
