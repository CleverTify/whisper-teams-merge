#!/bin/sh
# Two jobs: make the bind mounts writable, then route between the web app and
# the CLI.
#
# Routing exists because `docker compose run app warmup` overrides a Dockerfile
# CMD but *not* its ENTRYPOINT. With `ENTRYPOINT ["python","-m","app.web.server"]`
# the argument was simply appended, app.web.server never reads sys.argv, and
# every documented CLI command quietly started a web server instead — no output,
# no error, no transcript. Dispatching here makes both forms work:
#
#   docker compose up                          -> the web app on :8080
#   docker compose run --rm app warmup         -> the CLI
#   docker compose run --rm app python -c ...  -> anything else, verbatim
set -e

APP_UID=1000
APP_GID=1000

# ffmpeg and tesseract parse fully untrusted media in here, which is the most
# plausible code-execution path in this project, so the work does not run as
# root. But ./cache, ./input and ./output are bind mounts created by Docker as
# root on the host, and an unprivileged process cannot write to them. So: start
# as root, hand those three to the app user, then drop.
#
# `find -not -user` rather than a blind `chown -R`: /cache holds ~15 GB of
# models and re-chowning all of it on every start would be pure waiting.
if [ "$(id -u)" = "0" ]; then
  for dir in /cache /work/input /work/output; do
    [ -d "$dir" ] || mkdir -p "$dir"
    find "$dir" \( -not -user "$APP_UID" -o -not -group "$APP_GID" \) \
         -exec chown "$APP_UID:$APP_GID" {} + 2>/dev/null || true
  done
  # setpriv is in util-linux, already present; no gosu to install and trust.
  exec setpriv --reuid="$APP_UID" --regid="$APP_GID" --init-groups \
       "$0" "$@"
fi

case "${1:-serve}" in
  serve|"")
    exec python -m app.web.server
    ;;
  warmup|transcribe|reexport)
    exec python -m app.cli "$@"
    ;;
  *)
    # Not one of ours: run it as given, so `python`, `sh`, `pytest` and friends
    # behave the way anyone would expect from a container.
    exec "$@"
    ;;
esac
