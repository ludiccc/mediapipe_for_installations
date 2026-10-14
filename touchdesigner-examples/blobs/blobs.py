# Blobs receiver — TouchDesigner.
#
# Run this script INSIDE TouchDesigner and it builds a small network, `mp_blobs`,
# that receives the OSC messages of mediapipe_osc.py for the **Blobs** detector
# (OpenCV shapes that stand out from the background — people seen from a ceiling
# camera, objects on a table, silhouettes):
#
#   mp_blobs/oscin1           OSC In DAT on port 5005, with Python callbacks that
#   mp_blobs/oscin1_callbacks parse the messages into two tables:
#   mp_blobs/blobs            one row per blob:  id x y vx vy speed xmin ymin xmax ymax
#                                                area angle length width col row rank
#                             (col, row = the cell of a 3 x 3 grid of zones; rank = 1 for the biggest)
#   mp_blobs/contours         16 rows per blob: id index x y
#   mp_blobs/blobs_chop       Script CHOP: the `blobs` table as channels (one sample per blob), tx/ty
#   mp_blobs/zones_chop       Script CHOP: 9 channels zone_0_0 ... zone_2_2 = how many blobs in each cell
#   mp_blobs/geo_shapes       Geometry COMP with a Script SOP: the contour of every blob, its bounding
#                             box, main axis and the 3 x 3 grid
#   mp_blobs/geo_centers      Geometry COMP: a circle instanced on every centroid
#   mp_blobs/cam1, render1, over1, out1   the picture (click out1 or render1 to see it)
#
# How to run it (any of these):
#   1. drag this file from the file browser into a TouchDesigner network (it becomes a
#      Text DAT), right-click the DAT > "Run Script";
#   2. or in the Textport (Alt+T):  exec(open(r'C:\path\to\blobs.py').read())
# Run it again to rebuild from scratch.
#
# Camera side:   python mediapipe_osc.py --blobs     (ceiling camera, empty scene at start;
#                                                     or python osc_simulator.py)
#
# Messages used (first argument = stable id of the blob):
#   /mp/blobs           count id id ...          blobs on screen, biggest first (every frame)
#   /mp/blob/enter      id
#   /mp/blob/leave      id
#   /mp/blob/center     id x y 0                 centroid
#   /mp/blob/bbox       id xmin ymin xmax ymax
#   /mp/blob/velocity   id vx vy speed
#   /mp/blob/info       id area angle length width    area = fraction of the image (0..1),
#                                                     angle = main axis in degrees (-90..90)
#   /mp/blob/contour    id 16 x (x y)            outline, from the topmost point

PORT = 5005
NAME = 'mp_blobs'

# ---------------------------------------------------------------------------
# 1. Callbacks of the OSC In DAT.
# ---------------------------------------------------------------------------
OSC_CALLBACKS = r'''# Callbacks of the OSC In DAT (mediapipe_osc.py, Blobs detector).
# /mp/blob/... messages update the tables `blobs` and `contours`.

BLOBS_COLS = ['id', 'x', 'y', 'vx', 'vy', 'speed', 'xmin', 'ymin', 'xmax', 'ymax',
              'area', 'angle', 'length', 'width', 'col', 'row', 'rank']
CONTOUR_COLS = ['id', 'index', 'x', 'y']
GRID = 3                                     # zones per side


def reset_tables():
	b = op('blobs')
	b.clear()
	b.appendRow(BLOBS_COLS)
	c = op('contours')
	c.clear()
	c.appendRow(CONTOUR_COLS)


# Tables are the state: one row per object, the id in the first column.  The
# first row (index 0) is the header, so searches start at row 1.  Cells are
# strings: compare with str(id).
def find_row(t, bid):
	sid = str(bid)
	for r in range(1, t.numRows):
		if t[r, 0].val == sid:
			return r
	return -1


def blob_row(bid):
	t = op('blobs')
	r = find_row(t, bid)
	if r < 0:
		t.appendRow([bid] + [0] * (len(BLOBS_COLS) - 1))
		r = t.numRows - 1
		print('blob', bid, 'entered')
	return t, r


def set_cells(t, r, names, values):
	for name, value in zip(names, values):
		t[r, BLOBS_COLS.index(name)] = value


# Delete every row of an id, from the bottom up so the indices of the rows still to visit do not shift.
def delete_rows(t, bid):
	sid = str(bid)
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
		for name in ('blobs_chop', 'zones_chop', 'geo_shapes/shapes'):
			o = op(name)
			if o:
				o.cook(force=True)
		return
	# /mp/blobs: count, then the ids biggest first -> written to the `rank` column (1 = biggest).
	if address == '/mp/blobs':
		# count, then the ids biggest first -> the rank column
		t = op('blobs')
		for k, bid in enumerate(args[1:1 + int(args[0])]):
			r = find_row(t, bid)
			if r > 0:
				t[r, BLOBS_COLS.index('rank')] = k + 1
		return
	if not address.startswith('/mp/blob/'):
		return                                   # poses, hands, faces ... are ignored here

	bid = int(args[0])
	if address == '/mp/blob/enter':
		blob_row(bid)
	elif address == '/mp/blob/leave':
		delete_rows(op('blobs'), bid)
		delete_rows(op('contours'), bid)
		print('blob', bid, 'left')
	# center: the centroid - its grid cell is compared with the stored one so a
	# position stream becomes a "changed zone" event.
	elif address == '/mp/blob/center':
		t, r = blob_row(bid)
		x, y = args[1], args[2]
		col = min(GRID - 1, max(0, int(x * GRID)))
		row = min(GRID - 1, max(0, int(y * GRID)))
		if t[r, BLOBS_COLS.index('col')].val != str(col) or t[r, BLOBS_COLS.index('row')].val != str(row):
			print('blob', bid, 'is now in zone column', col, ', row', row)     # an event
		set_cells(t, r, ['x', 'y', 'col', 'row'], [x, y, col, row])
	elif address == '/mp/blob/bbox':
		t, r = blob_row(bid)
		set_cells(t, r, ['xmin', 'ymin', 'xmax', 'ymax'], args[1:5])
	elif address == '/mp/blob/velocity':
		t, r = blob_row(bid)
		set_cells(t, r, ['vx', 'vy', 'speed'], args[1:4])
	elif address == '/mp/blob/info':
		t, r = blob_row(bid)
		set_cells(t, r, ['area', 'angle', 'length', 'width'], args[1:5])
	elif address == '/mp/blob/contour':
		n = (len(args) - 1) // 2
		if n < 3:
			return
		c = op('contours')
		delete_rows(c, bid)
		for i in range(n):
			c.appendRow([bid, i, args[1 + 2 * i], args[2 + 2 * i]])
	return
'''

# ---------------------------------------------------------------------------
# 2. Script CHOP `blobs_chop`: the `blobs` table as channels, one sample per blob.
# ---------------------------------------------------------------------------
BLOBS_CHOP = r'''# Script CHOP: the `blobs` table as channels (one sample per blob) + tx, ty in render units.

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
	t = op('blobs')
	names = [c.val for c in t.row(0)]
	n = t.numRows - 1
	scriptOp.numSamples = max(1, n)
	scriptOp.appendChan('count').vals = [n] * max(1, n)
	asp = aspect()
	columns = {name: [] for name in names}
	tx, ty = [], []
	for r in range(1, t.numRows):
		for c, name in enumerate(names):
			try:
				columns[name].append(float(t[r, c].val))
			except ValueError:
				columns[name].append(0.0)
		tx.append((columns['x'][-1] - 0.5) * 2 * asp)
		ty.append((0.5 - columns['y'][-1]) * 2)
	for name, values in columns.items():
		scriptOp.appendChan(name).vals = values or [0.0]
	scriptOp.appendChan('tx').vals = tx or [0.0]
	scriptOp.appendChan('ty').vals = ty or [0.0]
	return
'''

# ---------------------------------------------------------------------------
# 3. Script CHOP `zones_chop`: how many blobs in each cell of the grid (9 channels, 1 sample).
# ---------------------------------------------------------------------------
ZONES_CHOP = r'''# Script CHOP: zone_<col>_<row> = number of blobs in that cell of the 3 x 3 grid.
# Drive anything "where the visitor stands" from these channels.

GRID = 3


# onCook runs every time the CHOP cooks (the OSC callbacks force one cook per
# frame).  scriptOp is this CHOP: clear it, set the number of samples, then
# appendChan(name) creates a channel whose .vals is a list of numSamples numbers.
# A CHOP cannot have 0 samples, so an empty table gives one sample of zeros.
def onCook(scriptOp):
	scriptOp.clear()
	scriptOp.numSamples = 1
	t = op('blobs')
	names = [c.val for c in t.row(0)]
	counts = {}
	for r in range(1, t.numRows):
		try:
			key = (int(t[r, names.index('col')].val), int(t[r, names.index('row')].val))
		except ValueError:
			continue
		counts[key] = counts.get(key, 0) + 1
	for col in range(GRID):
		for row in range(GRID):
			scriptOp.appendChan('zone_%d_%d' % (col, row)).vals = [float(counts.get((col, row), 0))]
	return
'''

# ---------------------------------------------------------------------------
# 4. Script SOP `shapes` (inside geo_shapes): contours, boxes, main axes and the grid.
# ---------------------------------------------------------------------------
SHAPES_SOP = r'''# Script SOP: the 3 x 3 grid, and for every blob its contour (closed), bounding box and main axis.
import math

GRID = 3


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
	# the grid
	for k in range(1, GRID):
		add_poly(scriptOp, [to_render(k / GRID, 0, asp), to_render(k / GRID, 1, asp)], closed=False)
		add_poly(scriptOp, [to_render(0, k / GRID, asp), to_render(1, k / GRID, asp)], closed=False)
	# contours
	c = op('../contours')
	pts = {}
	for r in range(1, c.numRows):
		bid, idx, x, y = [float(c[r, k].val) for k in range(4)]
		pts.setdefault(int(bid), []).append((int(idx), to_render(x, y, asp)))
	for bid, lst in pts.items():
		lst.sort()
		add_poly(scriptOp, [p for i, p in lst], closed=True)
	# boxes and main axes
	t = op('../blobs')
	names = [cell.val for cell in t.row(0)]
	for r in range(1, t.numRows):
		row = {name: t[r, k].val for k, name in enumerate(names)}
		try:
			x0, y0 = to_render(float(row['xmin']), float(row['ymin']), asp)
			x1, y1 = to_render(float(row['xmax']), float(row['ymax']), asp)
			cx, cy = to_render(float(row['x']), float(row['y']), asp)
			ang = math.radians(float(row['angle']))
			half = float(row['length']) / 2
		except ValueError:
			continue
		add_poly(scriptOp, [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], closed=False)
		dx, dy = math.cos(ang) * half * 2 * asp, -math.sin(ang) * half * 2
		add_poly(scriptOp, [(cx - dx, cy - dy), (cx + dx, cy + dy)], closed=False)
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

	blobs = c.create(tableDAT, 'blobs')
	blobs.nodeX, blobs.nodeY = 250, 0
	blobs.clear()
	blobs.appendRow(['id', 'x', 'y', 'vx', 'vy', 'speed', 'xmin', 'ymin', 'xmax', 'ymax',
	                 'area', 'angle', 'length', 'width', 'col', 'row', 'rank'])
	contours = c.create(tableDAT, 'contours')
	contours.nodeX, contours.nodeY = 250, -150
	contours.clear()
	contours.appendRow(['id', 'index', 'x', 'y'])

	# One Script CHOP per table, each with its callbacks DAT (the onCook code above).
	for name, text, y in (('blobs_chop', BLOBS_CHOP, 0), ('zones_chop', ZONES_CHOP, -150)):
		cbd = c.create(textDAT, name + '_callbacks')
		cbd.text = text
		cbd.nodeX, cbd.nodeY = 500, y - 75
		chop = c.create(scriptCHOP, name)
		chop.nodeX, chop.nodeY = 500, y
		setpar(chop, 'callbacks', name + '_callbacks')

	# Rendering.  A Geometry COMP draws the SOP inside it (render + display flags
	# on); with instancing on, the SOP is repeated once per sample of the Instance
	# OP, translated by the tx / ty channels.  The default torus is removed first.
	mat = c.create(constantMAT, 'mat_shapes')
	mat.nodeX, mat.nodeY = 750, -300
	setpar(mat, 'colorr', 1.0)
	setpar(mat, 'colorg', 0.5)
	setpar(mat, 'colorb', 0.2)

	geo = c.create(geometryCOMP, 'geo_shapes')
	geo.nodeX, geo.nodeY = 750, 0
	for child in list(geo.children):
		child.destroy()
	scb = geo.create(textDAT, 'shapes_callbacks')
	scb.text = SHAPES_SOP
	scb.nodeX, scb.nodeY = 0, -150
	shapes = geo.create(scriptSOP, 'shapes')
	setpar(shapes, 'callbacks', 'shapes_callbacks')
	shapes.render = True
	shapes.display = True
	setpar(geo, 'material', 'mat_shapes')

	geo2 = c.create(geometryCOMP, 'geo_centers')
	geo2.nodeX, geo2.nodeY = 750, -150
	for child in list(geo2.children):
		child.destroy()
	circle = geo2.create(circleSOP, 'circle1')
	setpar(circle, 'radx', 0.03)
	setpar(circle, 'rady', 0.03)
	circle.render = True
	circle.display = True
	setpar(geo2, 'material', 'mat_shapes')
	setpar(geo2, 'instancing', True)
	setpar(geo2, 'instanceop', 'blobs_chop')
	setpar(geo2, 'instancetx', 'tx')
	setpar(geo2, 'instancety', 'ty')

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
	setpar(render, 'geometry', 'geo_shapes geo_centers')
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

	print('built %s: OSC In DAT on port %d -> tables blobs / contours -> blobs_chop, zones_chop, render1' % (c.path, PORT))
	print('start the camera side:  python mediapipe_osc.py --blobs   (or python osc_simulator.py)')
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
