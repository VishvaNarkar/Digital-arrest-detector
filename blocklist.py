# blocklist.py
import json, os
BLOCKLIST_FILE = "data/blocklist.json"

def load_blocklist():
    if os.path.exists(BLOCKLIST_FILE):
        with open(BLOCKLIST_FILE) as f:
            return json.load(f)
    return {"domains":[], "phones":[]}

def add_to_blocklist(kind, value):
    bl = load_blocklist()
    bl.setdefault(kind+"s", [])
    if value not in bl[kind+"s"]:
        bl[kind+"s"].append(value)
        with open(BLOCKLIST_FILE, "w") as f:
            json.dump(bl, f)

def is_blocked(domain=None, phone=None):
    bl = load_blocklist()
    if domain and domain in bl.get("domains",[]):
        return True
    if phone and phone in bl.get("phones",[]):
        return True
    return False
