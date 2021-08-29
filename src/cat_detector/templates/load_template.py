from jinja2 import Template
import os

def load_template(name: str):
    path = os.path.join(os.path.dirname(__file__), f"{name}.jinja")
    with open(path, "rt") as f:
        return Template(f.read())