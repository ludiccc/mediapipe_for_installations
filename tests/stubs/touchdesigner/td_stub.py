"""A tiny stand-in for TouchDesigner's Python environment.

Used only by tests/test_examples.py to run the build scripts of
touchdesigner-examples/ outside TouchDesigner: operators are plain Python
objects, parameters accept anything, tables / Script CHOP / Script SOP keep the
data in lists, and the OSC In DAT can be fed messages directly.  It mimics the
small part of the TouchDesigner API the examples use — nothing more.
"""

from __future__ import annotations

import types


# ---------------------------------------------------------------- parameters
class Par:
    def __init__(self, value=None):
        self.val = value

    def eval(self):
        return self.val


class ParCollection:
    def __init__(self):
        object.__setattr__(self, "_pars", {})

    def __setattr__(self, name, value):
        self._pars[name] = value

    def __getattr__(self, name):
        if name in self._pars:
            return Par(self._pars[name])
        raise AttributeError(name)

    def __getitem__(self, name):
        return Par(self._pars.get(name))


# ---------------------------------------------------------------- operators
class Connector:
    def __init__(self, owner, index):
        self.owner, self.index, self.connected = owner, index, None

    def connect(self, other):
        self.connected = other
        self.owner.inputs[self.index] = other


class OP:
    OPType = "OP"
    family = "OP"

    def __init__(self, parent, name):
        self._parent = parent
        self.name = name
        self.children = []
        self.par = ParCollection()
        self.nodeX = self.nodeY = 0
        self.viewer = self.render = self.display = False
        self.comment = ""
        self.inputs = {}
        self.inputConnectors = [Connector(self, i) for i in range(4)]
        self.outputConnectors = [Connector(self, 0)]
        self.width, self.height = 1280, 720
        self.cooks = 0
        self._storage = {}

    @property
    def path(self):
        if self._parent is None:
            return "/"
        base = self._parent.path
        return ("" if base == "/" else base) + "/" + self.name

    def parent(self, n=1):
        p = self
        for _ in range(n):
            p = p._parent
        return p

    def op(self, path):
        return resolve(self, path)

    def create(self, cls, name):
        # creating an op with an existing name: TouchDesigner renames; we fail loudly
        if any(ch.name == name for ch in self.children):
            raise ValueError(f"{self.path} already has a child named {name}")
        child = cls(self, name)
        self.children.append(child)
        if cls is GeometryCOMP:                         # like TouchDesigner: a default torus inside
            child.children.append(SOP(child, "torus1"))
        return child

    def destroy(self):
        if self._parent:
            self._parent.children.remove(self)

    def cook(self, force=False, recurse=False):
        self.cooks += 1

    def store(self, key, value):
        self._storage[key] = value

    def fetch(self, key, default=None):
        return self._storage.get(key, default)

    def __repr__(self):
        return f"<{self.OPType} {self.path}>"

    def __bool__(self):
        return True


def resolve(start, path):
    """op(path) like TouchDesigner: absolute, or relative to the inside of a COMP /
    to the parent of any other operator (a script in a DAT sees its siblings by name)."""
    if path.startswith("/"):
        node = root()
        parts = [p for p in path.split("/") if p]
    else:
        node = start if isinstance(start, COMP) or start._parent is None else start.parent()
        parts = path.split("/")
    for part in parts:
        if part == "..":
            node = node.parent()
        elif part in ("", "."):
            continue
        else:
            found = [c for c in node.children if c.name == part]
            if not found:
                return None
            node = found[0]
    return node


class COMP(OP):
    OPType = "baseCOMP"
    family = "COMP"


class GeometryCOMP(COMP):
    OPType = "geometryCOMP"


class CameraCOMP(COMP):
    OPType = "cameraCOMP"


class ContainerCOMP(COMP):
    OPType = "containerCOMP"


class TOP(OP):
    OPType = "TOP"
    family = "TOP"


class RenderTOP(TOP):
    OPType = "renderTOP"


class ConstantTOP(TOP):
    OPType = "constantTOP"


class OverTOP(TOP):
    OPType = "overTOP"


class OutTOP(TOP):
    OPType = "outTOP"


class TextTOP(TOP):
    OPType = "textTOP"


class MAT(OP):
    OPType = "constantMAT"
    family = "MAT"


class SOP(OP):
    OPType = "SOP"
    family = "SOP"


class CircleSOP(SOP):
    OPType = "circleSOP"


# ---------------------------------------------------------------- DATs
class Cell:
    def __init__(self, table, row, col):
        self.table, self.row, self.col = table, row, col

    @property
    def val(self):
        return self.table.rows[self.row][self.col]

    @val.setter
    def val(self, v):
        self.table.rows[self.row][self.col] = str(v)

    def __repr__(self):
        return f"Cell({self.val!r})"


class DAT(OP):
    OPType = "textDAT"
    family = "DAT"

    def __init__(self, parent, name):
        super().__init__(parent, name)
        self.text = ""
        self._module = None

    def module(self):
        """The DAT's text executed as a module (with op() relative to the DAT, like TouchDesigner)."""
        if self._module is None:
            ns = {"op": lambda p: resolve(self, p), "me": self, "print": print, "__name__": self.name, "absTime": absTime}
            exec(compile(self.text, self.path, "exec"), ns)
            self._module = types.SimpleNamespace(**ns)
        return self._module


class TableDAT(DAT):
    OPType = "tableDAT"

    def __init__(self, parent, name):
        super().__init__(parent, name)
        self.rows = [["0"]]

    def clear(self, keepSize=False, keepFirstRow=False, keepFirstCol=False):
        self.rows = [self.rows[0]] if keepFirstRow else []

    def appendRow(self, values):
        self.rows.append([str(v) for v in values])
        return len(self.rows) - 1

    def appendRows(self, rows):
        for r in rows:
            self.appendRow(r)
        return len(self.rows) - 1

    def deleteRow(self, index):
        del self.rows[index]

    @property
    def numRows(self):
        return len(self.rows)

    @property
    def numCols(self):
        return max((len(r) for r in self.rows), default=0)

    def row(self, index):
        if isinstance(index, str):
            index = [r[0] for r in self.rows].index(index)
        return [Cell(self, index, c) for c in range(len(self.rows[index]))]

    def col(self, index):
        if isinstance(index, str):
            index = self.rows[0].index(index)
        return [Cell(self, r, index) for r in range(len(self.rows))]

    def _index(self, r, c):
        if isinstance(r, str):
            r = [row[0] for row in self.rows].index(r)
        if isinstance(c, str):
            c = self.rows[0].index(c)
        return r, c

    def __getitem__(self, key):
        r, c = self._index(*key)
        return Cell(self, r, c)

    def __setitem__(self, key, value):
        r, c = self._index(*key)
        self.rows[r][c] = str(value)


class OscInDAT(TableDAT):
    OPType = "oscinDAT"

    def receive(self, address, args):
        """What TouchDesigner does when a datagram arrives: call onReceiveOSC of the callbacks DAT."""
        cb = resolve(self, self.par.callbacks.eval())
        self.rows.append([address] + [str(a) for a in args])
        return cb.module().onReceiveOSC(self, len(self.rows) - 1, address, b"", 0.0, address, list(args), None)


# ---------------------------------------------------------------- Script CHOP / SOP
class Chan:
    def __init__(self, name):
        self.name = name
        self.vals = []

    def __getitem__(self, i):
        return self.vals[i]


class ScriptCHOP(OP):
    OPType = "scriptCHOP"
    family = "CHOP"

    def __init__(self, parent, name):
        super().__init__(parent, name)
        self.chans = []
        self.numSamples = 1

    def clear(self):
        self.chans = []

    def appendChan(self, name):
        ch = Chan(name)
        self.chans.append(ch)
        return ch

    def chan(self, name):
        for ch in self.chans:
            if ch.name == name:
                return ch
        return None

    def __getitem__(self, name):
        return self.chan(name)

    @property
    def numChans(self):
        return len(self.chans)

    def cook(self, force=False, recurse=False):
        super().cook(force)
        cb = resolve(self, self.par.callbacks.eval())
        cb.module().onCook(self)
        for ch in self.chans:
            if len(ch.vals) != self.numSamples:
                raise ValueError(f"{self.path}: channel {ch.name} has {len(ch.vals)} values, numSamples is {self.numSamples}")


class Point:
    def __init__(self):
        self.x = self.y = self.z = 0.0

    @property
    def P(self):
        return (self.x, self.y, self.z)

    @P.setter
    def P(self, v):
        self.x, self.y, self.z = v


class Vertex:
    def __init__(self):
        self.point = Point()


class Poly(list):
    def __init__(self, n, closed):
        super().__init__(Vertex() for _ in range(n))
        self.closed = closed


class ScriptSOP(SOP):
    OPType = "scriptSOP"

    def __init__(self, parent, name):
        super().__init__(parent, name)
        self.prims = []

    def clear(self):
        self.prims = []

    def appendPoly(self, n, closed=True, addPoints=True):
        p = Poly(n, closed)
        self.prims.append(p)
        return p

    @property
    def numPrims(self):
        return len(self.prims)

    def cook(self, force=False, recurse=False):
        super().cook(force)
        cb = resolve(self, self.par.callbacks.eval())
        cb.module().onCook(self)


class _AbsTime:
    """absTime.frame / absTime.seconds, driven by the test."""
    frame = 0
    seconds = 0.0


absTime = _AbsTime()


# ---------------------------------------------------------------- the world
_root = None


def root():
    return _root


def new_project():
    """A fresh root with /project1, like a new .toe."""
    global _root
    _root = COMP(None, "")
    _root.name = ""
    _root.children = []
    project1 = ContainerCOMP(_root, "project1")
    _root.children.append(project1)
    return project1


def globals_for(me=None):
    """The names the build scripts expect from TouchDesigner (class names, op(), me)."""
    return {
        "op": lambda p: resolve(me, p) if me is not None else resolve(_root, p.lstrip("/")),
        "me": me,
        "baseCOMP": COMP, "containerCOMP": ContainerCOMP, "geometryCOMP": GeometryCOMP, "cameraCOMP": CameraCOMP,
        "textDAT": DAT, "tableDAT": TableDAT, "oscinDAT": OscInDAT,
        "scriptCHOP": ScriptCHOP, "scriptSOP": ScriptSOP, "circleSOP": CircleSOP,
        "renderTOP": RenderTOP, "constantTOP": ConstantTOP, "overTOP": OverTOP, "outTOP": OutTOP, "textTOP": TextTOP,
        "absTime": absTime,
        "constantMAT": MAT,
    }
