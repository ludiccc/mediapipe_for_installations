#!/usr/bin/env python3
"""Convert a Processing sketch (.pde) to a Java class for the stub harness.

Does the few things Processing's preprocessor does that matter here:
* hoists the imports and wraps the code in ``public class <Name> extends PApplet``
* gives decimal literals an ``f`` suffix (Processing floats are 32-bit)
* ``color`` as a type becomes ``int``, ``#RRGGBB`` literals become ints
"""
import re
import sys


def convert(src: str, name: str) -> str:
    imports, body = [], []
    for line in src.splitlines():
        if re.match(r"\s*import\s+[\w.*]+\s*;", line):
            imports.append(line.strip())
        else:
            body.append(line)
    code = "\n".join(body)

    def fix_floats(segment):
        segment = re.sub(r"(?<![\w.])(\d+\.\d*|\.\d+)(?![\w.])", r"\1f", segment)
        return segment

    # keep string literals untouched
    parts = re.split(r'("(?:\\.|[^"\\])*")', code)
    code = "".join(p if i % 2 else fix_floats(p) for i, p in enumerate(parts))
    code = re.sub(r"\bcolor\s+(?=\w+\s*(=|;|,|\[))", "int ", code)              # color c = ...
    code = re.sub(r"\bcolor\[\]", "int[]", code)
    code = re.sub(r"#([0-9A-Fa-f]{6})\b", lambda m: str(int(m.group(1), 16) | 0xFF000000 - (1 << 32)), code)
    # top-level methods (column 0) are public in Processing
    code = re.sub(r"^(?!public|private|static|class|final)([\w\[\]<>]+\s+\w+\s*\()", r"public \1", code, flags=re.M)
    header = "import processing.core.*;\n" + "\n".join(imports) + "\n"
    header += "import java.util.HashMap;\nimport java.util.ArrayList;\n"
    return f"{header}\npublic class {name} extends PApplet {{\n{code}\n}}\n"


if __name__ == "__main__":
    pde, name, out = sys.argv[1:4]
    open(out, "w").write(convert(open(pde).read(), name))
