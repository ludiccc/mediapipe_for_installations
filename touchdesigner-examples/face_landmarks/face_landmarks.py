# Face landmarks receiver — TouchDesigner.
#
# Run this script INSIDE TouchDesigner and it builds a small network, `mp_face`,
# that receives the OSC messages of mediapipe_osc.py for the **Face** detector:
#
#   mp_face/oscin1            OSC In DAT on port 5005, with Python callbacks that
#   mp_face/oscin1_callbacks  parse the messages into three tables:
#   mp_face/faces             one row per face:  id x y z vx vy speed xmin ymin xmax ymax roll mouth_open
#   mp_face/blendshapes       one row per face:  id + the 52 blendshapes by name (jawOpen, eyeBlinkLeft ...)
#   mp_face/landmarks         478 rows per face: id index x y z   (only with --face-landmarks)
#   mp_face/faces_chop        Script CHOP: faces + blendshapes as channels (one sample per face), tx/ty
#   mp_face/landmarks_chop    Script CHOP: one sample per mesh point, tx/ty in render units
#   mp_face/geo_points        Geometry COMP: a circle instanced on every mesh point
#   mp_face/geo_lines         Geometry COMP with a Script SOP: bounding box, face oval, eye line, and
#                             an ellipse "mask" rotated with the head — where your own image would go
#   mp_face/cam1, render1, over1, out1   the picture (click out1 or render1 to see it)
#
# How to run it (any of these):
#   1. drag this file from the file browser into a TouchDesigner network (it becomes a
#      Text DAT), right-click the DAT > "Run Script";
#   2. or in the Textport (Alt+T):  exec(open(r'C:\path\to\face_landmarks.py').read())
# Run it again to rebuild from scratch.
#
# Camera side:   python mediapipe_osc.py --face --face-landmarks --mirror     (or python osc_simulator.py)
#
# Messages used (first argument = stable id of the face):
#   /mp/face/enter            id
#   /mp/face/leave            id
#   /mp/face/center           id x y z                 the nose tip
#   /mp/face/bbox             id xmin ymin xmax ymax
#   /mp/face/velocity         id vx vy speed
#   /mp/face/blendshapes      id 52 x value            0..1; order = /mp/face/blendshape_names
#   /mp/face/blendshape_names 52 x name                sent once when the app starts
#   /mp/face/landmarks        id 478 x (x y z)         only with --face-landmarks (a 7 KB message)
# Landmark indices: https://developers.google.com/mediapipe/solutions/vision/face_landmarker

PORT = 5005
NAME = 'mp_face'

BLENDSHAPE_NAMES = [
	'_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft',
	'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft',
	'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight',
	'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft',
	'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen',
	'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft',
	'mouthFrownRight', 'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight',
	'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower',
	'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight',
	'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight',
	'noseSneerLeft', 'noseSneerRight']

# ---------------------------------------------------------------------------
# 1. Callbacks of the OSC In DAT.
# ---------------------------------------------------------------------------
OSC_CALLBACKS = r'''# Callbacks of the OSC In DAT (mediapipe_osc.py, Face detector).
# /mp/face/... messages update the tables `faces`, `blendshapes` and `landmarks`.
import math

FACES_COLS = ['id', 'x', 'y', 'z', 'vx', 'vy', 'speed', 'xmin', 'ymin', 'xmax', 'ymax', 'roll', 'mouth_open']
LM_COLS = ['id', 'index', 'x', 'y', 'z']
MOUTH_OPEN = 0.4                             # jawOpen above this = "mouth open"
LEFT_EYE_OUTER, RIGHT_EYE_OUTER = 263, 33    # for the head roll


def names():
	"""The blendshape names = the header of the `blendshapes` table (minus the id column)."""
	return [c.val for c in op('blendshapes').row(0)][1:]


# The 478-point mesh is big: instead of editing rows one by one, the mesh of
# every face is cached here and the whole `landmarks` table is rebuilt in one
# appendRows call (rebuild_landmarks) whenever a face's mesh changes.
MESH = {}                                    # id -> 478 rows, the source of the `landmarks` table


def reset_tables():
	f = op('faces')
	f.clear()
	f.appendRow(FACES_COLS)
	b = op('blendshapes')
	header = [c.val for c in b.row(0)]
	b.clear()
	b.appendRow(header)
	MESH.clear()
	rebuild_landmarks()


def rebuild_landmarks():
	"""The `landmarks` table = the header + the cached mesh of every face (one call, it is big)."""
	l = op('landmarks')
	l.clear()
	l.appendRow(LM_COLS)
	rows = [row for fid in sorted(MESH) for row in MESH[fid]]
	if not rows:
		return
	if hasattr(l, 'appendRows'):
		l.appendRows(rows)
	else:
		for row in rows:
			l.appendRow(row)


# Tables are the state: one row per object, the id in the first column.  The
# first row (index 0) is the header, so searches start at row 1.  Cells are
# strings: compare with str(id).
def find_row(t, fid):
	sid = str(fid)
	for r in range(1, t.numRows):
		if t[r, 0].val == sid:
			return r
	return -1


def face_row(fid):
	t = op('faces')
	r = find_row(t, fid)
	if r < 0:
		t.appendRow([fid] + [0] * (len(FACES_COLS) - 1))
		r = t.numRows - 1
		print('face', fid, 'entered')
	return t, r


def set_cells(t, r, cols, values):
	for name, value in zip(cols, values):
		t[r, FACES_COLS.index(name)] = value


# Delete every row of an id, from the bottom up so the indices of the rows still to visit do not shift.
def delete_rows(t, fid):
	sid = str(fid)
	for r in range(t.numRows - 1, 0, -1):
		if t[r, 0].val == sid:
			t.deleteRow(r)


# TouchDesigner calls onReceiveOSC for every message the OSC In DAT receives:
# `address` is the OSC address, `args` the list of arguments (int / float / str).
# The first argument of every /mp/<kind>/... message is the stable id.
def onReceiveOSC(dat, rowIndex, message, bytes, timeStamp, address, args, peer):
	# /mp/status "started" | "stopped": on "stopped" the tables are emptied, because
	# no more leave messages will come.
	if address == '/mp/status':
		print('mediapipe_osc.py', args[0])
		if args[0] == 'stopped':
			reset_tables()
		return
	# /mp/frame arrives once per analysed camera frame: a good moment to make the
	# Script CHOPs / SOP recook with the data the previous messages stored.
	if address == '/mp/frame':
		for name in ('faces_chop', 'landmarks_chop', 'geo_lines/lines'):
			o = op(name)
			if o:
				o.cook(force=True)
		return
	if address == '/mp/face/blendshape_names':
		b = op('blendshapes')                    # the header of the table = the names, as sent
		b.clear()
		b.appendRow(['id'] + [str(a) for a in args])
		return
	if not address.startswith('/mp/face/'):
		return                                   # poses, hands, blobs ... are ignored here

	fid = int(args[0])
	if address == '/mp/face/enter':
		face_row(fid)
	elif address == '/mp/face/leave':
		for name in ('faces', 'blendshapes'):
			delete_rows(op(name), fid)
		if MESH.pop(fid, None) is not None:
			rebuild_landmarks()
		print('face', fid, 'left')
	elif address == '/mp/face/center':
		t, r = face_row(fid)
		set_cells(t, r, ['x', 'y', 'z'], args[1:4])
	elif address == '/mp/face/bbox':
		t, r = face_row(fid)
		set_cells(t, r, ['xmin', 'ymin', 'xmax', 'ymax'], args[1:5])
	elif address == '/mp/face/velocity':
		t, r = face_row(fid)
		set_cells(t, r, ['vx', 'vy', 'speed'], args[1:4])
	# blendshapes: one row per face in the `blendshapes` table, the 52 values under
	# their names (the header).  Then the jawOpen threshold gives the mouth event.
	elif address == '/mp/face/blendshapes':
		b = op('blendshapes')
		r = find_row(b, fid)
		values = [fid] + list(args[1:])
		if r < 0:
			b.appendRow(values)
		else:
			for c, v in enumerate(values):
				b[r, c] = v
		# an event: the mouth opens / closes
		n = names()
		jaw = args[1 + n.index('jawOpen')] if 'jawOpen' in n and len(args) > n.index('jawOpen') + 1 else 0
		t, r = face_row(fid)
		was_open = t[r, FACES_COLS.index('mouth_open')].val == '1'
		is_open = jaw > MOUTH_OPEN
		if is_open != was_open:
			print('face', fid, 'mouth', 'open' if is_open else 'closed')
			set_cells(t, r, ['mouth_open'], [1 if is_open else 0])
	# landmarks: cache the 478 rows of this face, rebuild the table, and compute the head roll from the eye corners.
	elif address == '/mp/face/landmarks':
		n = (len(args) - 1) // 3
		if n < 468:
			return
		MESH[fid] = [[fid, i, args[1 + i * 3], args[2 + i * 3], args[3 + i * 3]] for i in range(n)]
		rebuild_landmarks()
		# head roll from the outer eye corners (degrees, 0 = upright)
		lx, ly = args[1 + LEFT_EYE_OUTER * 3], args[2 + LEFT_EYE_OUTER * 3]
		rx, ry = args[1 + RIGHT_EYE_OUTER * 3], args[2 + RIGHT_EYE_OUTER * 3]
		t, r = face_row(fid)
		set_cells(t, r, ['roll'], [round(math.degrees(math.atan2(ly - ry, lx - rx)), 1)])
	return
'''

# ---------------------------------------------------------------------------
# 2. Script CHOP `faces_chop`: faces + blendshapes as channels, one sample per face.
# ---------------------------------------------------------------------------
FACES_CHOP = r'''# Script CHOP: the `faces` and `blendshapes` tables as channels (one sample per face) + tx, ty.

# The render is 1280 x 720 (16:9): its aspect ratio is needed because the
# orthographic camera shows x from -aspect to +aspect and y from -1 to 1.
def aspect():
	r = op('render1')
	return (r.width / r.height) if (r and r.height) else 16 / 9


# onCook runs every time the CHOP cooks (the OSC callbacks force one cook per
# frame).  scriptOp is this CHOP: clear it, set the number of samples, then
# appendChan(name) creates a channel whose .vals is a list of numSamples numbers.
# A CHOP cannot have 0 samples, so an empty table gives one sample of zeros.
def onCook(scriptOp):
	scriptOp.clear()
	t = op('faces')
	b = op('blendshapes')
	names = [c.val for c in t.row(0)]
	bnames = [c.val for c in b.row(0)][1:]
	n = t.numRows - 1
	scriptOp.numSamples = max(1, n)
	scriptOp.appendChan('count').vals = [n] * max(1, n)
	asp = aspect()
	columns = {name: [] for name in names}
	shapes = {name: [] for name in bnames}
	tx, ty = [], []
	for r in range(1, t.numRows):
		for c, name in enumerate(names):
			try:
				columns[name].append(float(t[r, c].val))
			except ValueError:
				columns[name].append(0.0)
		tx.append((columns['x'][-1] - 0.5) * 2 * asp)
		ty.append((0.5 - columns['y'][-1]) * 2)
		# the blendshapes of this face (0 if not received yet)
		br = -1
		for k in range(1, b.numRows):
			if b[k, 0].val == t[r, 0].val:
				br = k
		for c, name in enumerate(bnames):
			try:
				shapes[name].append(float(b[br, c + 1].val) if br > 0 else 0.0)
			except ValueError:
				shapes[name].append(0.0)
	for name, values in columns.items():
		scriptOp.appendChan(name).vals = values or [0.0]
	scriptOp.appendChan('tx').vals = tx or [0.0]
	scriptOp.appendChan('ty').vals = ty or [0.0]
	for name, values in shapes.items():
		scriptOp.appendChan(name).vals = values or [0.0]
	return
'''

# ---------------------------------------------------------------------------
# 3. Script CHOP `landmarks_chop`: one sample per mesh point (all faces).
# ---------------------------------------------------------------------------
LANDMARKS_CHOP = r'''# Script CHOP: the `landmarks` table as channels (one sample per mesh point) + tx, ty.

# The render is 1280 x 720 (16:9): its aspect ratio is needed because the
# orthographic camera shows x from -aspect to +aspect and y from -1 to 1.
def aspect():
	r = op('render1')
	return (r.width / r.height) if (r and r.height) else 16 / 9


# onCook runs every time the CHOP cooks (the OSC callbacks force one cook per
# frame).  scriptOp is this CHOP: clear it, set the number of samples, then
# appendChan(name) creates a channel whose .vals is a list of numSamples numbers.
# A CHOP cannot have 0 samples, so an empty table gives one sample of zeros.
def onCook(scriptOp):
	scriptOp.clear()
	t = op('landmarks')
	n = t.numRows - 1
	scriptOp.numSamples = max(1, n)
	asp = aspect()
	cols = {name: [] for name in ('id', 'index', 'x', 'y', 'z', 'tx', 'ty')}
	for r in range(1, t.numRows):
		fid, idx, x, y, z = [float(t[r, c].val) for c in range(5)]
		cols['id'].append(fid)
		cols['index'].append(idx)
		cols['x'].append(x)
		cols['y'].append(y)
		cols['z'].append(z)
		cols['tx'].append((x - 0.5) * 2 * asp)
		cols['ty'].append((0.5 - y) * 2)
	for name, values in cols.items():
		scriptOp.appendChan(name).vals = values or [0.0]
	return
'''

# ---------------------------------------------------------------------------
# 4. Script SOP `lines` (inside geo_lines): box, oval, eye line and the rotated "mask".
# ---------------------------------------------------------------------------
LINES_SOP = r'''# Script SOP: for every face, its bounding box, the face oval and the eye line (from the
# mesh, when received) and an ellipse "mask" the size of the box, rotated with the head —
# where an installation would place its own image.
import math

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
	152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_EYE_OUTER, RIGHT_EYE_OUTER = 263, 33


# The render is 1280 x 720 (16:9): its aspect ratio is needed because the
# orthographic camera shows x from -aspect to +aspect and y from -1 to 1.
def aspect():
	r = op('../render1')
	return (r.width / r.height) if (r and r.height) else 16 / 9


def to_render(x, y, asp):
	return (x - 0.5) * 2 * asp, (0.5 - y) * 2


def add_poly(scriptOp, pts, closed):
	poly = scriptOp.appendPoly(len(pts), closed=closed, addPoints=True)
	for k, (x, y) in enumerate(pts):
		poly[k].point.x, poly[k].point.y = x, y


# onCook runs every time the SOP cooks.  scriptOp.appendPoly(n, closed, addPoints)
# creates a polygon with n new points; poly[k].point.x / .y place them.  Open
# polygons render as lines, closed ones as filled shapes (Constant MAT, no light).
def onCook(scriptOp):
	scriptOp.clear()
	asp = aspect()
	faces = op('../faces')
	names = [c.val for c in faces.row(0)]
	t = op('../landmarks')
	points = {}
	for r in range(1, t.numRows):
		fid, idx, x, y, z = [float(t[r, c].val) for c in range(5)]
		points[(int(fid), int(idx))] = to_render(x, y, asp)
	for r in range(1, faces.numRows):
		row = {name: faces[r, c].val for c, name in enumerate(names)}
		try:
			fid = int(row['id'])
			x0, y0 = to_render(float(row['xmin']), float(row['ymin']), asp)
			x1, y1 = to_render(float(row['xmax']), float(row['ymax']), asp)
			roll = math.radians(float(row['roll']))
		except ValueError:
			continue
		add_poly(scriptOp, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], closed=False)   # the box (outline)
		# the mask: an ellipse the size of the box, rotated by the head roll (y is up here: -roll)
		cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, abs(x1 - x0) / 2, abs(y1 - y0) / 2
		pts = []
		for k in range(32):
			a = 2 * math.pi * k / 32
			ex, ey = rx * math.cos(a), ry * math.sin(a)
			pts.append((cx + ex * math.cos(-roll) - ey * math.sin(-roll), cy + ex * math.sin(-roll) + ey * math.cos(-roll)))
		add_poly(scriptOp, pts, closed=True)
		# from the mesh: the oval and the eye line
		oval = [points[(fid, i)] for i in FACE_OVAL if (fid, i) in points]
		if len(oval) == len(FACE_OVAL):
			add_poly(scriptOp, oval, closed=False)
			add_poly(scriptOp, [points[(fid, LEFT_EYE_OUTER)], points[(fid, RIGHT_EYE_OUTER)]], closed=False)
	return
'''


# ---------------------------------------------------------------------------
# the builder
# ---------------------------------------------------------------------------
# Everything below runs once, when the script is executed inside TouchDesigner.
# It creates the operators with comp.create(type, name), sets parameters with
# op.par.<name> = value, and wires the TOPs with inputConnectors[i].connect().
# setpar() never lets a refused parameter stop the build: it prints a WARNING
# instead, so a renamed parameter in another TouchDesigner version only costs
# one manual setting.
def build():
	warnings = []

	def setpar(o, name, value):
		try:
			setattr(o.par, name, value)
		except Exception as e:
			warnings.append('%s: could not set %s = %r (%s) -> set it by hand' % (o.path, name, value, e))

	try:
		# Where to build: next to the Text DAT that runs the script (drag & drop), or in
		# /project1 when run from the Textport (`me` is not a DAT there).
		where = me.parent() if me.OPType.endswith('DAT') else None
	except Exception:
		where = None
	if where is None:
		where = op('/project1') or op('/')

	# Rebuild from scratch: an older copy with the same name is deleted first.
	old = where.op(NAME)
	if old:
		old.destroy()
	c = where.create(baseCOMP, NAME)
	c.nodeX, c.nodeY = 0, 0

	cb = c.create(textDAT, 'oscin1_callbacks')
	cb.text = OSC_CALLBACKS
	cb.nodeX, cb.nodeY = 0, -150
	# The OSC In DAT listens on PORT; `callbacks` points to the Text DAT whose
	# onReceiveOSC parses the messages.  Names are relative to the same component,
	# so 'oscin1_callbacks' is enough.
	osc = c.create(oscinDAT, 'oscin1')
	setpar(osc, 'port', PORT)
	setpar(osc, 'callbacks', 'oscin1_callbacks')
	setpar(osc, 'active', True)

	faces = c.create(tableDAT, 'faces')
	faces.nodeX, faces.nodeY = 250, 0
	faces.clear()
	faces.appendRow(['id', 'x', 'y', 'z', 'vx', 'vy', 'speed', 'xmin', 'ymin', 'xmax', 'ymax', 'roll', 'mouth_open'])
	shapes = c.create(tableDAT, 'blendshapes')
	shapes.nodeX, shapes.nodeY = 250, -150
	shapes.clear()
	shapes.appendRow(['id'] + BLENDSHAPE_NAMES)
	landmarks = c.create(tableDAT, 'landmarks')
	landmarks.nodeX, landmarks.nodeY = 250, -300
	landmarks.clear()
	landmarks.appendRow(['id', 'index', 'x', 'y', 'z'])

	# One Script CHOP per table, each with its callbacks DAT (the onCook code above).
	for name, text, y in (('faces_chop', FACES_CHOP, 0), ('landmarks_chop', LANDMARKS_CHOP, -150)):
		cbd = c.create(textDAT, name + '_callbacks')
		cbd.text = text
		cbd.nodeX, cbd.nodeY = 500, y - 75
		chop = c.create(scriptCHOP, name)
		chop.nodeX, chop.nodeY = 500, y
		setpar(chop, 'callbacks', name + '_callbacks')

	# Rendering.  A Geometry COMP draws the SOP inside it (render + display flags
	# on); with instancing on, the SOP is repeated once per sample of the Instance
	# OP, translated by the tx / ty channels.  The default torus is removed first.
	mat = c.create(constantMAT, 'mat_points')
	mat.nodeX, mat.nodeY = 750, -300
	setpar(mat, 'colorr', 0.4)
	setpar(mat, 'colorg', 0.9)
	setpar(mat, 'colorb', 1.0)

	geo = c.create(geometryCOMP, 'geo_points')
	geo.nodeX, geo.nodeY = 750, 0
	for child in list(geo.children):
		child.destroy()
	circle = geo.create(circleSOP, 'circle1')
	setpar(circle, 'radx', 0.006)
	setpar(circle, 'rady', 0.006)
	circle.render = True
	circle.display = True
	setpar(geo, 'material', 'mat_points')
	setpar(geo, 'instancing', True)
	setpar(geo, 'instanceop', 'landmarks_chop')
	setpar(geo, 'instancetx', 'tx')
	setpar(geo, 'instancety', 'ty')

	geo2 = c.create(geometryCOMP, 'geo_lines')
	geo2.nodeX, geo2.nodeY = 750, -150
	for child in list(geo2.children):
		child.destroy()
	lcb = geo2.create(textDAT, 'lines_callbacks')
	lcb.text = LINES_SOP
	lcb.nodeX, lcb.nodeY = 0, -150
	lines = geo2.create(scriptSOP, 'lines')
	setpar(lines, 'callbacks', 'lines_callbacks')
	lines.render = True
	lines.display = True
	setpar(geo2, 'material', 'mat_points')

	# Camera, Render TOP, a dark background composited under the render (a Render
	# TOP alone is transparent where nothing is drawn) and an Out TOP to look at.
	# The camera is orthographic with width 2 * aspect: x -aspect..aspect, y -1..1,
	# which is what the tx / ty channels assume.
	cam = c.create(cameraCOMP, 'cam1')
	cam.nodeX, cam.nodeY = 750, 150
	render = c.create(renderTOP, 'render1')
	render.nodeX, render.nodeY = 1000, 0
	setpar(render, 'resolutionw', 1280)
	setpar(render, 'resolutionh', 720)
	setpar(render, 'camera', 'cam1')
	setpar(render, 'geometry', 'geo_points geo_lines')
	setpar(cam, 'projection', 'ortho')
	setpar(cam, 'orthowidth', 2 * 1280 / 720)

	bg = c.create(constantTOP, 'background')
	bg.nodeX, bg.nodeY = 1000, -150
	setpar(bg, 'colorr', 0.08)
	setpar(bg, 'colorg', 0.08)
	setpar(bg, 'colorb', 0.08)
	setpar(bg, 'resolutionw', 1280)
	setpar(bg, 'resolutionh', 720)
	over = c.create(overTOP, 'over1')
	over.nodeX, over.nodeY = 1250, 0
	out = c.create(outTOP, 'out1')
	out.nodeX, out.nodeY = 1500, 0
	try:
		over.inputConnectors[0].connect(render)
		over.inputConnectors[1].connect(bg)
		out.inputConnectors[0].connect(over)
	except Exception as e:
		warnings.append('could not wire render1 -> over1 -> out1 (%s): connect them by hand' % e)
	try:
		out.viewer = True
		c.par.opviewer = 'out1'
	except Exception:
		pass

	print('built %s: OSC In DAT on port %d -> tables faces / blendshapes / landmarks -> faces_chop, landmarks_chop, render1' % (c.path, PORT))
	print('start the camera side:  python mediapipe_osc.py --face --face-landmarks --mirror   (or python osc_simulator.py)')
	for w in warnings:
		print('WARNING', w)
	return c


try:
	op
# The last lines: in TouchDesigner `op` exists as a builtin, so build() runs;
# in plain Python it does not, so the script only prints a hint.
except NameError:
	print('This script must be run inside TouchDesigner (drag it into a network, right-click > Run Script).')
else:
	build()
