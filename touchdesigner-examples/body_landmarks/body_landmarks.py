# Body landmarks (pose) receiver — TouchDesigner.
#
# Run this script INSIDE TouchDesigner and it builds a small network, `mp_body`,
# that receives the OSC messages of mediapipe_osc.py for the **Body** detector:
#
#   mp_body/oscin1            OSC In DAT on port 5005, with Python callbacks that
#   mp_body/oscin1_callbacks  parse the messages into two tables:
#   mp_body/people            one row per person:  id x y z vx vy speed xmin ymin xmax ymax zone hand_up
#   mp_body/landmarks         33 rows per person:  id index x y z visibility
#   mp_body/people_chop       Script CHOP: the `people` table as channels (one sample per person)
#   mp_body/landmarks_chop    Script CHOP: the `landmarks` table as channels (one sample per point),
#                             with tx/ty in render units — used to instance a circle per landmark
#   mp_body/geo_points        Geometry COMP: a circle instanced on landmarks_chop
#   mp_body/geo_skeleton      Geometry COMP with a Script SOP that draws the skeleton lines
#   mp_body/cam1, render1, over1, out1   the picture (click out1 or render1 to see it)
#
# How to run it (any of these):
#   1. drag this file from the file browser into a TouchDesigner network (it becomes a
#      Text DAT), right-click the DAT > "Run Script";
#   2. or paste the whole file in the Textport (Alt+T) — it builds into /project1.
# Run it again to rebuild from scratch (mp_body is deleted and created again).
#
# Camera side:   python mediapipe_osc.py --pose --mirror     (or python osc_simulator.py)
#
# Messages used (first argument = stable id of the person):
#   /mp/pose/enter      id
#   /mp/pose/leave      id
#   /mp/pose/center     id x y z                  visibility-weighted centre of the body
#   /mp/pose/bbox       id xmin ymin xmax ymax
#   /mp/pose/velocity   id vx vy speed            normalized units per second
#   /mp/pose/landmarks  id 33 x (x y z visibility)
# x, y are normalized (0..1, origin top-left).  Landmark indices:
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

PORT = 5005
NAME = 'mp_body'

# ---------------------------------------------------------------------------
# 1. Callbacks of the OSC In DAT: every message updates the tables.
#    (If you build the network by hand, paste this text in the callbacks DAT.)
# ---------------------------------------------------------------------------
OSC_CALLBACKS = r'''# Callbacks of the OSC In DAT (mediapipe_osc.py, Body detector).
# Every /mp/pose/... message updates the tables `people` and `landmarks`.

PEOPLE_COLS = ['id', 'x', 'y', 'z', 'vx', 'vy', 'speed', 'xmin', 'ymin', 'xmax', 'ymax', 'zone', 'hand_up']
LM_COLS = ['id', 'index', 'x', 'y', 'z', 'visibility']
L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 11, 12, 15, 16


def reset_tables():
	p = op('people')
	p.clear()
	p.appendRow(PEOPLE_COLS)
	l = op('landmarks')
	l.clear()
	l.appendRow(LM_COLS)


# Tables are the state: one row per object, the id in the first column.  The
# first row (index 0) is the header, so searches start at row 1.  Cells are
# strings: compare with str(id).
def find_row(t, pid):
	"""Row index of the person `pid` in a table whose first column is the id, or -1."""
	sid = str(pid)
	for r in range(1, t.numRows):
		if t[r, 0].val == sid:
			return r
	return -1


def person_row(pid):
	"""Row of the person in `people`, created with zeros if it is new."""
	t = op('people')
	r = find_row(t, pid)
	if r < 0:
		t.appendRow([pid] + [0] * (len(PEOPLE_COLS) - 1))
		r = t.numRows - 1
		print('person', pid, 'entered')
	return t, r


def set_cells(t, r, names, values):
	for name, value in zip(names, values):
		t[r, PEOPLE_COLS.index(name)] = value


# Delete every row of an id, from the bottom up so the indices of the rows still to visit do not shift.
def delete_rows(t, pid):
	sid = str(pid)
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
		# once per analysis frame: make sure the CHOPs / SOP recook with the new data
		for name in ('people_chop', 'landmarks_chop', 'geo_skeleton/skeleton'):
			o = op(name)
			if o:
				o.cook(force=True)
		return
	if not address.startswith('/mp/pose/'):
		return                                   # hands, faces, blobs ... are ignored here

	pid = int(args[0])
	if address == '/mp/pose/enter':
		person_row(pid)
	elif address == '/mp/pose/leave':
		delete_rows(op('people'), pid)
		delete_rows(op('landmarks'), pid)
		print('person', pid, 'left')
	# center: x y z, plus the zone (which third of the picture, 0 left .. 2 right).
	elif address == '/mp/pose/center':
		t, r = person_row(pid)
		x, y, z = args[1], args[2], args[3]
		zone = min(2, max(0, int(x * 3)))        # 0 = left, 1 = centre, 2 = right third
		set_cells(t, r, ['x', 'y', 'z', 'zone'], [x, y, z, zone])
	elif address == '/mp/pose/bbox':
		t, r = person_row(pid)
		set_cells(t, r, ['xmin', 'ymin', 'xmax', 'ymax'], args[1:5])
	elif address == '/mp/pose/velocity':
		t, r = person_row(pid)
		set_cells(t, r, ['vx', 'vy', 'speed'], args[1:4])
	# landmarks: 33 rows for this person are replaced (delete + append), then the
	# "hand up" flag is computed: a wrist above the shoulders (smaller y).
	elif address == '/mp/pose/landmarks':
		# 33 x (x y z visibility) after the id
		if len(args) < 1 + 33 * 4:
			return
		l = op('landmarks')
		delete_rows(l, pid)
		for i in range(33):
			l.appendRow([pid, i, args[1 + i * 4], args[2 + i * 4], args[3 + i * 4], args[4 + i * 4]])
		# a wrist above the shoulders (y grows downwards) -> hand_up
		shoulders = min(args[2 + L_SHOULDER * 4], args[2 + R_SHOULDER * 4])
		up = (args[2 + L_WRIST * 4] < shoulders and args[4 + L_WRIST * 4] > 0.5) or \
		     (args[2 + R_WRIST * 4] < shoulders and args[4 + R_WRIST * 4] > 0.5)
		t, r = person_row(pid)
		set_cells(t, r, ['hand_up'], [1 if up else 0])
	return
'''

# ---------------------------------------------------------------------------
# 2. Script CHOP `people_chop`: the `people` table as channels, one sample per person,
#    plus tx/ty = the centre in render units (the camera is orthographic, height 2).
# ---------------------------------------------------------------------------
PEOPLE_CHOP = r'''# Script CHOP: the `people` table as channels (one sample per person) + tx, ty in render units.

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
	t = op('people')
	names = [c.val for c in t.row(0)]
	n = t.numRows - 1
	scriptOp.numSamples = max(1, n)
	count = scriptOp.appendChan('count')
	count.vals = [n] * max(1, n)
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
	for name in names:
		chan = scriptOp.appendChan(name)
		chan.vals = columns[name] or [0.0]
	scriptOp.appendChan('tx').vals = tx or [0.0]
	scriptOp.appendChan('ty').vals = ty or [0.0]
	return
'''

# ---------------------------------------------------------------------------
# 3. Script CHOP `landmarks_chop`: one sample per landmark (all people), with tx/ty
#    for instancing and `vis` for the size.
# ---------------------------------------------------------------------------
LANDMARKS_CHOP = r'''# Script CHOP: the `landmarks` table as channels (one sample per landmark) + tx, ty in render units.

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
	cols = {name: [] for name in ('id', 'index', 'x', 'y', 'z', 'visibility', 'tx', 'ty', 'size')}
	for r in range(1, t.numRows):
		pid, idx, x, y, z, vis = [float(t[r, c].val) for c in range(6)]
		cols['id'].append(pid)
		cols['index'].append(idx)
		cols['x'].append(x)
		cols['y'].append(y)
		cols['z'].append(z)
		cols['visibility'].append(vis)
		cols['tx'].append((x - 0.5) * 2 * asp)
		cols['ty'].append((0.5 - y) * 2)
		cols['size'].append(0.5 + vis)           # instance scale: more visible = bigger
	for name, values in cols.items():
		scriptOp.appendChan(name).vals = values or [0.0]
	return
'''

# ---------------------------------------------------------------------------
# 4. Script SOP `skeleton` (inside geo_skeleton): one line per connection of every person.
# ---------------------------------------------------------------------------
SKELETON_SOP = r'''# Script SOP: the skeleton (MediaPipe's POSE_CONNECTIONS) of every person in `landmarks`.

CONNECTIONS = [(0, 1), (0, 4), (1, 2), (2, 3), (3, 7), (4, 5), (5, 6), (6, 8), (9, 10),
	(11, 12), (11, 13), (11, 23), (12, 14), (12, 24), (13, 15), (14, 16),
	(15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22), (17, 19), (18, 20),
	(23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (27, 31), (28, 30), (28, 32),
	(29, 31), (30, 32)]


# The render is 1280 x 720 (16:9): its aspect ratio is needed because the
# orthographic camera shows x from -aspect to +aspect and y from -1 to 1.
def aspect():
	r = op('../render1')
	return (r.width / r.height) if (r and r.height) else 16 / 9


# onCook runs every time the SOP cooks.  scriptOp.appendPoly(n, closed, addPoints)
# creates a polygon with n new points; poly[k].point.x / .y place them.  Open
# polygons render as lines, closed ones as filled shapes (Constant MAT, no light).
def onCook(scriptOp):
	scriptOp.clear()
	t = op('../landmarks')                       # the table lives one level up, in mp_body
	asp = aspect()
	points = {}                                  # (id, index) -> (tx, ty, visibility)
	for r in range(1, t.numRows):
		pid, idx, x, y, z, vis = [float(t[r, c].val) for c in range(6)]
		points[(int(pid), int(idx))] = ((x - 0.5) * 2 * asp, (0.5 - y) * 2, vis)
	ids = sorted({pid for pid, idx in points})
	for pid in ids:
		for a, b in CONNECTIONS:
			pa, pb = points.get((pid, a)), points.get((pid, b))
			if pa is None or pb is None or pa[2] < 0.3 or pb[2] < 0.3:
				continue
			poly = scriptOp.appendPoly(2, closed=False, addPoints=True)
			poly[0].point.x, poly[0].point.y, poly[0].point.z = pa[0], pa[1], 0
			poly[1].point.x, poly[1].point.y, poly[1].point.z = pb[0], pb[1], 0
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
		"""Set a parameter; if TouchDesigner refuses, remember it and go on."""
		try:
			setattr(o.par, name, value)
		except Exception as e:
			warnings.append('%s: could not set %s = %r (%s) -> set it by hand' % (o.path, name, value, e))

	# where: next to the DAT that runs this script, or in /project1
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
	try:
		c.viewer = True
	except Exception:
		pass

	# --- OSC In DAT + callbacks + tables ------------------------------------
	cb = c.create(textDAT, 'oscin1_callbacks')
	cb.text = OSC_CALLBACKS
	cb.nodeX, cb.nodeY = 0, -150
	# The OSC In DAT listens on PORT; `callbacks` points to the Text DAT whose
	# onReceiveOSC parses the messages.  Names are relative to the same component,
	# so 'oscin1_callbacks' is enough.
	osc = c.create(oscinDAT, 'oscin1')
	osc.nodeX, osc.nodeY = 0, 0
	setpar(osc, 'port', PORT)
	setpar(osc, 'callbacks', 'oscin1_callbacks')
	setpar(osc, 'active', True)

	people = c.create(tableDAT, 'people')
	people.nodeX, people.nodeY = 250, 0
	people.clear()
	people.appendRow(['id', 'x', 'y', 'z', 'vx', 'vy', 'speed', 'xmin', 'ymin', 'xmax', 'ymax', 'zone', 'hand_up'])
	landmarks = c.create(tableDAT, 'landmarks')
	landmarks.nodeX, landmarks.nodeY = 250, -150
	landmarks.clear()
	landmarks.appendRow(['id', 'index', 'x', 'y', 'z', 'visibility'])

	# --- Script CHOPs --------------------------------------------------------
	# One Script CHOP per table, each with its callbacks DAT (the onCook code above).
	for name, text, y in (('people_chop', PEOPLE_CHOP, 0), ('landmarks_chop', LANDMARKS_CHOP, -150)):
		cbd = c.create(textDAT, name + '_callbacks')
		cbd.text = text
		cbd.nodeX, cbd.nodeY = 500, y - 75
		chop = c.create(scriptCHOP, name)
		chop.nodeX, chop.nodeY = 500, y
		setpar(chop, 'callbacks', name + '_callbacks')

	# --- geometry: circles instanced on the landmarks, lines for the skeleton ---
	# Rendering.  A Geometry COMP draws the SOP inside it (render + display flags
	# on); with instancing on, the SOP is repeated once per sample of the Instance
	# OP, translated by the tx / ty channels.  The default torus is removed first.
	mat = c.create(constantMAT, 'mat_points')
	mat.nodeX, mat.nodeY = 750, -300
	setpar(mat, 'colorr', 0.3)
	setpar(mat, 'colorg', 1.0)
	setpar(mat, 'colorb', 0.4)

	geo = c.create(geometryCOMP, 'geo_points')
	geo.nodeX, geo.nodeY = 750, 0
	for child in list(geo.children):
		child.destroy()
	circle = geo.create(circleSOP, 'circle1')
	setpar(circle, 'radx', 0.02)
	setpar(circle, 'rady', 0.02)
	circle.render = True
	circle.display = True
	setpar(geo, 'material', 'mat_points')
	setpar(geo, 'instancing', True)
	setpar(geo, 'instanceop', 'landmarks_chop')
	setpar(geo, 'instancetx', 'tx')
	setpar(geo, 'instancety', 'ty')
	setpar(geo, 'instancesx', 'size')
	setpar(geo, 'instancesy', 'size')
	setpar(geo, 'instancesz', 'size')

	geo2 = c.create(geometryCOMP, 'geo_skeleton')
	geo2.nodeX, geo2.nodeY = 750, -150
	for child in list(geo2.children):
		child.destroy()
	skcb = geo2.create(textDAT, 'skeleton_callbacks')
	skcb.text = SKELETON_SOP
	skcb.nodeX, skcb.nodeY = 0, -150
	sk = geo2.create(scriptSOP, 'skeleton')
	setpar(sk, 'callbacks', 'skeleton_callbacks')
	sk.render = True
	sk.display = True
	setpar(geo2, 'material', 'mat_points')

	# --- camera, render, black background, output ----------------------------
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
	setpar(render, 'geometry', 'geo_points geo_skeleton')
	# orthographic camera: x = -aspect..aspect, y = -1..1 (what the Script CHOPs assume)
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

	print('built %s: OSC In DAT on port %d -> tables people / landmarks -> people_chop, landmarks_chop, render1' % (c.path, PORT))
	print('start the camera side:  python mediapipe_osc.py --pose --mirror   (or python osc_simulator.py)')
	for w in warnings:
		print('WARNING', w)
	return c


try:
	# The last lines: in TouchDesigner `op` exists as a builtin, so build() runs;
	# in plain Python it does not, so the script only prints a hint.
	op                                             # TouchDesigner's builtin
except NameError:
	print('This script must be run inside TouchDesigner (drag it into a network, right-click > Run Script).')
else:
	build()
