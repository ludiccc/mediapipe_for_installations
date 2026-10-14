# Hand landmarks receiver — TouchDesigner.
#
# Run this script INSIDE TouchDesigner and it builds a small network, `mp_hands`,
# that receives the OSC messages of mediapipe_osc.py for the **Hands** detector:
#
#   mp_hands/oscin1           OSC In DAT on port 5005, with Python callbacks that
#   mp_hands/oscin1_callbacks parse the messages into two tables:
#   mp_hands/hands            one row per hand:  id handedness score x y z vx vy speed
#                                                 xmin ymin xmax ymax pinch pinching
#   mp_hands/landmarks        21 rows per hand:  id index x y z
#   mp_hands/hands_chop       Script CHOP: the `hands` table as channels (one sample per hand;
#                             handedness as `right` = 1/0)
#   mp_hands/landmarks_chop   Script CHOP: one sample per landmark, tx/ty in render units
#   mp_hands/geo_points       Geometry COMP: a circle instanced on landmarks_chop (fingertips bigger)
#   mp_hands/geo_skeleton     Geometry COMP with a Script SOP: the hand connections + the pinch circle
#   mp_hands/cam1, render1, over1, out1   the picture (click out1 or render1 to see it)
#
# How to run it (any of these):
#   1. drag this file from the file browser into a TouchDesigner network (it becomes a
#      Text DAT), right-click the DAT > "Run Script";
#   2. or in the Textport (Alt+T):  exec(open(r'C:\path\to\hand_landmarks.py').read())
# Run it again to rebuild from scratch (mp_hands is deleted and created again).
#
# Camera side:   python mediapipe_osc.py --hands --mirror     (or python osc_simulator.py)
#
# Messages used (first argument = stable id of the hand):
#   /mp/hand/enter      id
#   /mp/hand/leave      id
#   /mp/hand/info       id handedness score       handedness = "Left" | "Right"
#   /mp/hand/center     id x y z
#   /mp/hand/bbox       id xmin ymin xmax ymax
#   /mp/hand/velocity   id vx vy speed
#   /mp/hand/pinch      id distance               thumb tip <-> index tip, roughly metres
#   /mp/hand/landmarks  id 21 x (x y z)
# Landmark indices: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker

PORT = 5005
NAME = 'mp_hands'

# ---------------------------------------------------------------------------
# 1. Callbacks of the OSC In DAT: every message updates the tables.
# ---------------------------------------------------------------------------
OSC_CALLBACKS = r'''# Callbacks of the OSC In DAT (mediapipe_osc.py, Hands detector).
# Every /mp/hand/... message updates the tables `hands` and `landmarks`.

HANDS_COLS = ['id', 'handedness', 'score', 'x', 'y', 'z', 'vx', 'vy', 'speed',
              'xmin', 'ymin', 'xmax', 'ymax', 'pinch', 'pinching']
LM_COLS = ['id', 'index', 'x', 'y', 'z']
PINCH_CLOSED = 0.04                          # pinch distance below this = "pinching"


def reset_tables():
	h = op('hands')
	h.clear()
	h.appendRow(HANDS_COLS)
	l = op('landmarks')
	l.clear()
	l.appendRow(LM_COLS)


# Tables are the state: one row per object, the id in the first column.  The
# first row (index 0) is the header, so searches start at row 1.  Cells are
# strings: compare with str(id).
def find_row(t, hid):
	sid = str(hid)
	for r in range(1, t.numRows):
		if t[r, 0].val == sid:
			return r
	return -1


def hand_row(hid):
	"""Row of the hand in `hands`, created if it is new."""
	t = op('hands')
	r = find_row(t, hid)
	if r < 0:
		t.appendRow([hid, '?', 0] + [0] * (len(HANDS_COLS) - 3))
		r = t.numRows - 1
		print('hand', hid, 'entered')
	return t, r


def set_cells(t, r, names, values):
	for name, value in zip(names, values):
		t[r, HANDS_COLS.index(name)] = value


# Delete every row of an id, from the bottom up so the indices of the rows still to visit do not shift.
def delete_rows(t, hid):
	sid = str(hid)
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
		for name in ('hands_chop', 'landmarks_chop', 'geo_skeleton/skeleton'):
			o = op(name)
			if o:
				o.cook(force=True)
		return
	if not address.startswith('/mp/hand/'):
		return                                   # poses, faces, blobs ... are ignored here

	hid = int(args[0])
	if address == '/mp/hand/enter':
		hand_row(hid)
	elif address == '/mp/hand/leave':
		delete_rows(op('hands'), hid)
		delete_rows(op('landmarks'), hid)
		print('hand', hid, 'left')
	elif address == '/mp/hand/info':
		t, r = hand_row(hid)
		set_cells(t, r, ['handedness', 'score'], [args[1], args[2]])
	elif address == '/mp/hand/center':
		t, r = hand_row(hid)
		set_cells(t, r, ['x', 'y', 'z'], args[1:4])
	elif address == '/mp/hand/bbox':
		t, r = hand_row(hid)
		set_cells(t, r, ['xmin', 'ymin', 'xmax', 'ymax'], args[1:5])
	elif address == '/mp/hand/velocity':
		t, r = hand_row(hid)
		set_cells(t, r, ['vx', 'vy', 'speed'], args[1:4])
	# pinch arrives every frame: compare with the threshold and print only when the
	# state changes - a continuous value becomes an event (pinch! / released).
	elif address == '/mp/hand/pinch':
		t, r = hand_row(hid)
		pinching = 1 if args[1] < PINCH_CLOSED else 0
		if str(pinching) != t[r, HANDS_COLS.index('pinching')].val:     # an event
			print('hand', hid, 'pinch!' if pinching else 'released')
		set_cells(t, r, ['pinch', 'pinching'], [args[1], pinching])
	# landmarks: the 21 rows of this hand are replaced (delete + append).
	elif address == '/mp/hand/landmarks':
		if len(args) < 1 + 21 * 3:
			return
		l = op('landmarks')
		delete_rows(l, hid)
		for i in range(21):
			l.appendRow([hid, i, args[1 + i * 3], args[2 + i * 3], args[3 + i * 3]])
	return
'''

# ---------------------------------------------------------------------------
# 2. Script CHOP `hands_chop`: the `hands` table as channels, one sample per hand.
# ---------------------------------------------------------------------------
HANDS_CHOP = r'''# Script CHOP: the `hands` table as channels (one sample per hand) + right (1/0), tx, ty.

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
	t = op('hands')
	names = [c.val for c in t.row(0)]
	n = t.numRows - 1
	scriptOp.numSamples = max(1, n)
	scriptOp.appendChan('count').vals = [n] * max(1, n)
	asp = aspect()
	columns = {name: [] for name in names if name != 'handedness'}
	right, tx, ty = [], [], []
	for r in range(1, t.numRows):
		for c, name in enumerate(names):
			if name == 'handedness':
				right.append(1.0 if t[r, c].val == 'Right' else 0.0)
				continue
			try:
				columns[name].append(float(t[r, c].val))
			except ValueError:
				columns[name].append(0.0)
		tx.append((columns['x'][-1] - 0.5) * 2 * asp)
		ty.append((0.5 - columns['y'][-1]) * 2)
	for name, values in columns.items():
		scriptOp.appendChan(name).vals = values or [0.0]
	scriptOp.appendChan('right').vals = right or [0.0]
	scriptOp.appendChan('tx').vals = tx or [0.0]
	scriptOp.appendChan('ty').vals = ty or [0.0]
	return
'''

# ---------------------------------------------------------------------------
# 3. Script CHOP `landmarks_chop`: one sample per landmark (all hands).
# ---------------------------------------------------------------------------
LANDMARKS_CHOP = r'''# Script CHOP: the `landmarks` table as channels (one sample per landmark) + tx, ty, size.

FINGERTIPS = (4, 8, 12, 16, 20)


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
	cols = {name: [] for name in ('id', 'index', 'x', 'y', 'z', 'tx', 'ty', 'size')}
	for r in range(1, t.numRows):
		hid, idx, x, y, z = [float(t[r, c].val) for c in range(5)]
		cols['id'].append(hid)
		cols['index'].append(idx)
		cols['x'].append(x)
		cols['y'].append(y)
		cols['z'].append(z)
		cols['tx'].append((x - 0.5) * 2 * asp)
		cols['ty'].append((0.5 - y) * 2)
		cols['size'].append(1.6 if int(idx) in FINGERTIPS else 0.8)     # fingertips bigger
	for name, values in cols.items():
		scriptOp.appendChan(name).vals = values or [0.0]
	return
'''

# ---------------------------------------------------------------------------
# 4. Script SOP `skeleton` (inside geo_skeleton): hand connections + a circle for the pinch.
# ---------------------------------------------------------------------------
SKELETON_SOP = r'''# Script SOP: the connections of every hand in `landmarks`, and a circle between thumb
# tip and index tip whose radius is the pinch distance.
import math

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
	(11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
THUMB_TIP, INDEX_TIP = 4, 8


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
	t = op('../landmarks')
	hands = op('../hands')
	asp = aspect()
	points = {}
	for r in range(1, t.numRows):
		hid, idx, x, y, z = [float(t[r, c].val) for c in range(5)]
		points[(int(hid), int(idx))] = ((x - 0.5) * 2 * asp, (0.5 - y) * 2)
	pinch = {}
	for r in range(1, hands.numRows):
		try:
			pinch[int(hands[r, 0].val)] = float(hands[r, 'pinch'].val)
		except ValueError:
			pass
	for hid in sorted({h for h, i in points}):
		for a, b in CONNECTIONS:
			pa, pb = points.get((hid, a)), points.get((hid, b))
			if pa is None or pb is None:
				continue
			poly = scriptOp.appendPoly(2, closed=False, addPoints=True)
			poly[0].point.x, poly[0].point.y = pa
			poly[1].point.x, poly[1].point.y = pb
		# the pinch circle
		pt, pi = points.get((hid, THUMB_TIP)), points.get((hid, INDEX_TIP))
		if pt and pi:
			cx, cy = (pt[0] + pi[0]) / 2, (pt[1] + pi[1]) / 2
			radius = max(0.01, pinch.get(hid, 0.1) * 0.8)
			poly = scriptOp.appendPoly(24, closed=True, addPoints=True)
			for k in range(24):
				a = 2 * math.pi * k / 24
				poly[k].point.x, poly[k].point.y = cx + radius * math.cos(a), cy + radius * math.sin(a)
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

	hands = c.create(tableDAT, 'hands')
	hands.nodeX, hands.nodeY = 250, 0
	hands.clear()
	hands.appendRow(['id', 'handedness', 'score', 'x', 'y', 'z', 'vx', 'vy', 'speed',
	                 'xmin', 'ymin', 'xmax', 'ymax', 'pinch', 'pinching'])
	landmarks = c.create(tableDAT, 'landmarks')
	landmarks.nodeX, landmarks.nodeY = 250, -150
	landmarks.clear()
	landmarks.appendRow(['id', 'index', 'x', 'y', 'z'])

	# One Script CHOP per table, each with its callbacks DAT (the onCook code above).
	for name, text, y in (('hands_chop', HANDS_CHOP, 0), ('landmarks_chop', LANDMARKS_CHOP, -150)):
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
	setpar(mat, 'colorr', 1.0)
	setpar(mat, 'colorg', 0.8)
	setpar(mat, 'colorb', 0.3)

	geo = c.create(geometryCOMP, 'geo_points')
	geo.nodeX, geo.nodeY = 750, 0
	for child in list(geo.children):
		child.destroy()
	circle = geo.create(circleSOP, 'circle1')
	setpar(circle, 'radx', 0.015)
	setpar(circle, 'rady', 0.015)
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

	print('built %s: OSC In DAT on port %d -> tables hands / landmarks -> hands_chop, landmarks_chop, render1' % (c.path, PORT))
	print('start the camera side:  python mediapipe_osc.py --hands --mirror   (or python osc_simulator.py)')
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
