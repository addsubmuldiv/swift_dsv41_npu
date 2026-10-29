#!/bin/sh
# git credential helper：仅对 host=gitcode.com 的 get 请求，从 scripts/.push_config.json
# （0600）读 token 回显。token 不进 argv/env/磁盘副本，仅存于 0600 配置。
[ "$1" = "get" ] || exit 0
HOST=""
while IFS= read -r line; do
    case "$line" in
        host=*) HOST="${line#host=}" ;;
    esac
    [ -z "$line" ] && break
done
[ "$HOST" = "gitcode.com" ] || exit 0
CFG="$(dirname "$0")/.push_config.json"
python3 - "$CFG" <<'PYEOF'
import json, sys
try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
    user = data["ci_repo"]["remote_url"].split("/")[-2]
    token = data["repo"]["api_token"]
except Exception:
    sys.exit(0)
print(f"username={user}")
print(f"password={token}")
print()
PYEOF
