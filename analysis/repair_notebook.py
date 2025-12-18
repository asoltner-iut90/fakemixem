import json

def extract_inner(nb):
    # Tant qu'on a un notebook avec une seule cellule raw
    while (
        isinstance(nb, dict)
        and "cells" in nb
        and len(nb["cells"]) == 1
        and nb["cells"][0].get("cell_type") == "raw"
    ):
        src = "".join(nb["cells"][0].get("source", []))
        src = src.strip()

        if not src.startswith("{"):
            break

        try:
            nb = json.loads(src)
        except json.JSONDecodeError:
            break

    return nb


with open("phase2.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

recovered = extract_inner(nb)

with open("phase2_recovered.ipynb", "w", encoding="utf-8") as f:
    json.dump(recovered, f, indent=2, ensure_ascii=False)

print("Notebook réparé : phase2_recovered.ipynb")