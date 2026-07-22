#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${PREFIX:-$ROOT/toolchains/llvm18}"
STATE="$PREFIX/.apt-state"
PACKAGES="$PREFIX/.packages"
KEYRING="$STATE/llvm-snapshot.gpg"
SOURCE_LIST="$STATE/sources.list"

mkdir -p "$STATE/lists/partial" "$STATE/archives/partial" "$PACKAGES" "$PREFIX/sysroot"
curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor --yes -o "$KEYRING"
printf 'deb [signed-by=%s] https://apt.llvm.org/focal/ llvm-toolchain-focal-18 main\n' \
  "$KEYRING" > "$SOURCE_LIST"

APT_OPTIONS=(
  -o "Dir::Etc::sourcelist=$SOURCE_LIST"
  -o "Dir::Etc::sourceparts=-"
  -o "Dir::State::lists=$STATE/lists"
  -o "Dir::Cache::archives=$STATE/archives"
  -o "APT::Get::List-Cleanup=0"
)

apt-get "${APT_OPTIONS[@]}" update
(
  cd "$PACKAGES"
  apt-get "${APT_OPTIONS[@]}" download \
    clang-18 \
    libclang-common-18-dev \
    libclang-cpp18 \
    libllvm18 \
    libomp-18-dev \
    libomp5-18 \
    llvm-18-linker-tools
)

for package in "$PACKAGES"/*.deb; do
  dpkg-deb -x "$package" "$PREFIX/sysroot"
done

CLANG="$PREFIX/sysroot/usr/lib/llvm-18/bin/clang"
if [[ ! -x "$CLANG" ]]; then
  echo "clang binary missing after extraction: $CLANG" >&2
  exit 1
fi

export LD_LIBRARY_PATH="$PREFIX/sysroot/usr/lib/x86_64-linux-gnu:$PREFIX/sysroot/usr/lib/llvm-18/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$CLANG" --version
"$CLANG" -march=znver4 -### -x c /dev/null -c 2>&1 | grep -- '-target-cpu' | head -n 1
sha256sum "$PACKAGES"/*.deb > "$PREFIX/packages.sha256"
du -sh "$PREFIX"

if [[ "${KEEP_DOWNLOADS:-0}" != "1" ]]; then
  rm -rf "$STATE" "$PACKAGES"
fi

printf '%s\n' "$CLANG"
