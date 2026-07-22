#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${PREFIX:-$ROOT/toolchains/gcc13}"
STATE="$PREFIX/.apt-state"
ARCHIVES="$STATE/archives"
KEYRING="$STATE/ubuntu-toolchain-r.gpg"
SOURCE_LIST="$STATE/sources.list"
FINGERPRINT="C8EC952E2A0E1FBDC5090F6A2C277A0A352154E5"
export GNUPGHOME="$STATE/gnupg"

mkdir -p "$STATE/lists/partial" "$ARCHIVES/partial" "$PREFIX/sysroot" "$GNUPGHOME"
chmod 700 "$GNUPGHOME"
curl -fsSL \
  "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x$FINGERPRINT" \
  | gpg --dearmor --yes -o "$KEYRING"
gpg --show-keys --with-colons "$KEYRING" | grep -q "^fpr:::::::::$FINGERPRINT:"
printf 'deb [signed-by=%s] https://ppa.launchpadcontent.net/ubuntu-toolchain-r/test/ubuntu focal main\n' \
  "$KEYRING" > "$SOURCE_LIST"

APT_OPTIONS=(
  -o "Dir::Etc::sourcelist=$SOURCE_LIST"
  -o "Dir::Etc::sourceparts=-"
  -o "Dir::State::lists=$STATE/lists"
  -o "Dir::Cache::archives=$ARCHIVES"
  -o "APT::Get::List-Cleanup=0"
)

apt-get "${APT_OPTIONS[@]}" update
apt-get "${APT_OPTIONS[@]}" -y --download-only --no-install-recommends install gcc-13

for package in "$ARCHIVES"/*.deb; do
  dpkg-deb -x "$package" "$PREFIX/sysroot"
done

GCC="$PREFIX/sysroot/usr/bin/gcc-13"
if [[ ! -x "$GCC" ]]; then
  echo "gcc binary missing after extraction: $GCC" >&2
  exit 1
fi

export GCC_EXEC_PREFIX="$PREFIX/sysroot/usr/lib/gcc/"
export COMPILER_PATH="$PREFIX/sysroot/usr/lib/gcc/x86_64-linux-gnu/13"
export LIBRARY_PATH="$PREFIX/sysroot/usr/lib/gcc/x86_64-linux-gnu/13:$PREFIX/sysroot/usr/lib/x86_64-linux-gnu"
"$GCC" --version
"$GCC" -march=znver4 -mtune=znver4 -Q --help=target -x c /dev/null \
  | grep -E '^  -m(arch|tune)=' | head -n 2
sha256sum "$ARCHIVES"/*.deb > "$PREFIX/packages.sha256"
du -sh "$PREFIX"

if [[ "${KEEP_DOWNLOADS:-0}" != "1" ]]; then
  rm -rf "$STATE"
fi

printf '%s\n' "$GCC"
