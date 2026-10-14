# Hand gestures receiver — TouchDesigner.
#
# Run this script INSIDE TouchDesigner and it builds a small network, `mp_gestures`,
# that receives the OSC messages of mediapipe_osc.py for the **Gestures** detector
# (hands + gesture classification) and turns them into things TouchDesigner can use:
#
#   mp_gestures/oscin1            OSC In DAT on port 5005, with Python callbacks that
#   mp_gestures/oscin1_callbacks  parse the messages into:
#   mp_gestures/hands             one row per hand:  id handedness gesture score x y xmin ymin xmax ymax
#   mp_gestures/events            the last 20 gesture *changes*:  time id handedness gesture score
#                                 (the events an installation reacts to)
#   mp_gestures/hands_chop        Script CHOP, one sample per hand: gesture as a number (`gesture_index`,
#                                 see GESTURES below), one 1/0 channel per gesture name (`open_palm`,
#                                 `thumb_up`...), score, x, y, tx, ty
#   mp_gestures/gesture_text      Text TOP that shows the gesture of the strongest hand, on a
#                                 background colour per gesture (set by the callbacks every frame)
#   mp_gestures/out1              the picture
#
# How to run it (any of these):
#   1. drag this file from the file browser into a TouchDesigner network (it becomes a
#      Text DAT), right-click the DAT > "Run Script";
#   2. or in the Textport (Alt+T):  exec(open(r'C:\path\to\hand_gestures.py').read())
# Run it again to rebuild from scratch.
#
# Camera side:   python mediapipe_osc.py --gesture --mirror     (or python osc_simulator.py)
#
# Messages used (first argument = stable id of the hand):
#   /mp/hand/enter      id
#   /mp/hand/leave      id
#   /mp/hand/info       id handedness score
#   /mp/hand/center     id x y z
#   /mp/hand/bbox       id xmin ymin xmax ymax
#   /mp/hand/gesture    id name score    name: None, Closed_Fist, Open_Palm, Pointing_Up,
#                                              Thumb_Down, Thumb_Up, Victory, ILoveYou

PORT = 5005
NAME = 'mp_gestures'

# ---------------------------------------------------------------------------
# 1. Callbacks of the OSC In DAT.
# ---------------------------------------------------------------------------
OSC_CALLBACKS = r'''# Callbacks of the OSC In DAT (mediapipe_osc.py, Gestures detector).
# /mp/hand/... messages update the table `hands`; a change of gesture is appended to `events`.

HANDS_COLS = ['id', 'handedness', 'gesture', 'score', 'x', 'y', 'xmin', 'ymin', 'xmax', 'ymax']
EVENTS_COLS = ['time', 'id', 'handedness', 'gesture', 'score']
MIN_SCORE = 0.5                              # ignore gestures the classifier is not sure about
MAX_EVENTS = 20


def reset_tables():
	h = op('hands')
	h.clear()
	h.appendRow(HANDS_COLS)


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
	t = op('hands')
	r = find_row(t, hid)
	if r < 0:
		t.appendRow([hid, '?', 'None', 0, 0, 0, 0, 0, 0, 0])
		r = t.numRows - 1
		print('hand', hid, 'entered')
	return t, r


def set_cells(t, r, names, values):
	for name, value in zip(names, values):
		t[r, HANDS_COLS.index(name)] = value


GESTURES = ['None', 'Closed_Fist', 'Open_Palm', 'Pointing_Up', 'Thumb_Down', 'Thumb_Up', 'Victory', 'ILoveYou']
COLORS = {'None': (0.1, 0.1, 0.1), 'Closed_Fist': (0.6, 0.2, 0.2), 'Open_Palm': (0.2, 0.5, 0.2),
          'Pointing_Up': (0.15, 0.4, 0.6), 'Thumb_Down': (0.4, 0.15, 0.4), 'Thumb_Up': (0.7, 0.6, 0.1),
          'Victory': (0.15, 0.5, 0.6), 'ILoveYou': (0.6, 0.2, 0.5)}


# Called from the /mp/frame handler: an example of *reacting* to the data by setting parameters of other operators.
def show_strongest_gesture():
	"""Once per frame: the Text TOP shows the gesture of the hand with the best score,
	on a background colour per gesture (the reaction of this example)."""
	text = op('gesture_text')
	if not text:
		return
	t = op('hands')
	best, best_score = 'no hands', -1.0
	for r in range(1, t.numRows):
		try:
			score = float(t[r, HANDS_COLS.index('score')].val)
		except ValueError:
			continue
		if score > best_score:
			best, best_score = t[r, HANDS_COLS.index('gesture')].val, score
	rgb = COLORS.get(best, (0.05, 0.05, 0.05))
	try:
		text.par.text = best
		text.par.bgcolorr, text.par.bgcolorg, text.par.bgcolorb = rgb
	except Exception:
		pass                                     # a parameter named differently in this version: no text


# Append a row to the `events` table (and keep it short): a DAT Execute DAT on that table is the cleanest trigger.
def log_event(hid, handedness, gesture, score):
	"""An event: the gesture of a hand changed.  React here (or watch the `events` table)."""
	print('hand', hid, '(%s):' % handedness, gesture, '(%.2f)' % score)
	e = op('events')
	e.appendRow([round(absTime.seconds, 2), hid, handedness, gesture, round(score, 2)])
	while e.numRows > 1 + MAX_EVENTS:
		e.deleteRow(1)


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
		o = op('hands_chop')
		if o:
			o.cook(force=True)
		show_strongest_gesture()
		return
	if not address.startswith('/mp/hand/'):
		return

	hid = int(args[0])
	if address == '/mp/hand/enter':
		hand_row(hid)
	elif address == '/mp/hand/leave':
		t = op('hands')
		r = find_row(t, hid)
		if r > 0:
			t.deleteRow(r)
		print('hand', hid, 'left')
	# The gesture arrives every frame; comparing it with the one stored in the table
	# turns the stream into events (a row in `events` only when it changes).
	elif address == '/mp/hand/gesture':
		# the interesting one: a gesture per hand, every frame -> events on change
		t, r = hand_row(hid)
		name, score = args[1], args[2]
		if score < MIN_SCORE:
			name = 'None'
		if name != t[r, HANDS_COLS.index('gesture')].val:
			log_event(hid, t[r, HANDS_COLS.index('handedness')].val, name, score)
		set_cells(t, r, ['gesture', 'score'], [name, score])
	elif address == '/mp/hand/info':
		t, r = hand_row(hid)
		set_cells(t, r, ['handedness'], [args[1]])
	elif address == '/mp/hand/center':
		t, r = hand_row(hid)
		set_cells(t, r, ['x', 'y'], args[1:3])
	elif address == '/mp/hand/bbox':
		t, r = hand_row(hid)
		set_cells(t, r, ['xmin', 'ymin', 'xmax', 'ymax'], args[1:5])
	return
'''

# ---------------------------------------------------------------------------
# 2. Script CHOP `hands_chop`: one sample per hand; the gesture as numbers.
# ---------------------------------------------------------------------------
HANDS_CHOP = r'''# Script CHOP: the `hands` table as channels (one sample per hand).
# gesture_index = position in GESTURES; one 1/0 channel per gesture (open_palm, thumb_up ...).

GESTURES = ['None', 'Closed_Fist', 'Open_Palm', 'Pointing_Up', 'Thumb_Down', 'Thumb_Up', 'Victory', 'ILoveYou']


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
	numeric = {name: [] for name in names if name not in ('handedness', 'gesture')}
	right, gesture_index, tx, ty = [], [], [], []
	flags = {g.lower(): [] for g in GESTURES}
	for r in range(1, t.numRows):
		row = {name: t[r, c].val for c, name in enumerate(names)}
		for name in numeric:
			try:
				numeric[name].append(float(row[name]))
			except ValueError:
				numeric[name].append(0.0)
		right.append(1.0 if row['handedness'] == 'Right' else 0.0)
		g = row['gesture'] if row['gesture'] in GESTURES else 'None'
		gesture_index.append(float(GESTURES.index(g)))
		for name in GESTURES:
			flags[name.lower()].append(1.0 if name == g else 0.0)
		tx.append((numeric['x'][-1] - 0.5) * 2 * asp)
		ty.append((0.5 - numeric['y'][-1]) * 2)
	for name, values in numeric.items():
		scriptOp.appendChan(name).vals = values or [0.0]
	scriptOp.appendChan('right').vals = right or [0.0]
	scriptOp.appendChan('gesture_index').vals = gesture_index or [0.0]
	for name, values in flags.items():
		scriptOp.appendChan(name).vals = values or [0.0]
	scriptOp.appendChan('tx').vals = tx or [0.0]
	scriptOp.appendChan('ty').vals = ty or [0.0]
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
	hands.appendRow(['id', 'handedness', 'gesture', 'score', 'x', 'y', 'xmin', 'ymin', 'xmax', 'ymax'])
	events = c.create(tableDAT, 'events')
	events.nodeX, events.nodeY = 250, -150
	events.clear()
	events.appendRow(['time', 'id', 'handedness', 'gesture', 'score'])

	# The Script CHOP with its callbacks DAT (the onCook code above).
	cbd = c.create(textDAT, 'hands_chop_callbacks')
	cbd.text = HANDS_CHOP
	cbd.nodeX, cbd.nodeY = 500, -75
	chop = c.create(scriptCHOP, 'hands_chop')
	chop.nodeX, chop.nodeY = 500, 0
	setpar(chop, 'callbacks', 'hands_chop_callbacks')

	# a Text TOP with the gesture of the strongest hand (an expression on the table)
	# The picture of this example is a Text TOP: the OSC callbacks write the gesture
	# name and the background colour into its parameters once per frame
	# (show_strongest_gesture), so no render network is needed.
	text = c.create(textTOP, 'gesture_text')
	text.nodeX, text.nodeY = 750, 0
	setpar(text, 'resolutionw', 1280)
	setpar(text, 'resolutionh', 720)
	setpar(text, 'fontsizex', 90)
	setpar(text, 'text', 'waiting for mediapipe_osc.py ...')
	setpar(text, 'bgalpha', 1)          # the callbacks set text + bgcolor once per frame
	out = c.create(outTOP, 'out1')
	out.nodeX, out.nodeY = 1000, 0
	try:
		out.inputConnectors[0].connect(text)
	except Exception as e:
		warnings.append('could not wire gesture_text -> out1 (%s)' % e)
	try:
		out.viewer = True
		c.par.opviewer = 'out1'
	except Exception:
		pass

	print('built %s: OSC In DAT on port %d -> tables hands / events -> hands_chop, gesture_text' % (c.path, PORT))
	print('start the camera side:  python mediapipe_osc.py --gesture --mirror   (or python osc_simulator.py)')
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
