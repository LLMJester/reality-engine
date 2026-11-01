"""A toy arm for builder plumbing tests: a key-value 'model' with a sum query."""


class ToyState:
    def __init__(self):
        self.rep = {}


class ToyArm:
    name = "TOY"
    custom_episode = False

    def tools(self):
        return [{"type": "function", "function": {"name": "set_value", "description": "Store a named number.",
                 "parameters": {"type": "object", "properties": {"name": {"type": "string"},
                                "value": {"type": "number"}}, "required": ["name", "value"]}}},
                {"type": "function", "function": {"name": "total", "description": "Sum of all stored numbers.",
                 "parameters": {"type": "object", "properties": {}}}}]

    current_tools = lambda self, st: self.tools()
    gold_tools = lambda self, st=None: [t for t in self.tools() if t["function"]["name"] == "total"]

    def new_state(self):
        return ToyState()

    def load(self, rep, given_gold=False):
        s = ToyState(); s.rep = dict(rep); return s

    def handle(self, st, name, args):
        if name == "set_value":
            try:
                st.rep[str(args["name"])] = float(args["value"]); return f"ok: {len(st.rep)} values"
            except Exception as e:
                return f"ERROR: {e}"
        if name == "total":
            return str(sum(st.rep.values()))
        return f"ERROR: unknown tool {name}"

    def export(self, st):
        return dict(st.rep)

    def validity(self, st):
        return {"usable": bool(st.rep), "executable": True, "errors": []}
