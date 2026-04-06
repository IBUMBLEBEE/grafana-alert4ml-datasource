.PHONY: all install-frontend install-go-deps build-rs-amd64 build-rs-arm64 build-go-amd64 build-go-arm64 build-ts docker-up clean

# ── Rust targets (musl static) ─────────────────────────────────
RUST_TARGET_AMD64 = x86_64-unknown-linux-musl
RUST_TARGET_ARM64 = aarch64-unknown-linux-musl

# ── Default: build amd64 ───────────────────────────────────────
all: install-frontend install-go-deps build-rs-amd64 build-rs-arm64 build-go-amd64 build-go-arm64 build-ts docker-up clean

# ── All platforms ──────────────────────────────────────────────
all-platforms: build-rs-amd64 build-rs-arm64 build-go-amd64 build-go-arm64 build-ts

install-frontend:
	npm install

install-go-deps:
	go mod download

# ── Rust builds ────────────────────────────────────────────────
build-rs-amd64:
	cd rsod && cargo zigbuild --release --target $(RUST_TARGET_AMD64)

build-rs-arm64:
	cd rsod && cargo zigbuild --release --target $(RUST_TARGET_ARM64)

# zig cc wrappers (auto-created, requires zig in PATH)
.build/x86_64-linux-musl-gcc:
	@mkdir -p .build
	@printf '#!/bin/sh\nexec zig cc -target x86_64-linux-musl "$$@"\n' > $@
	@chmod +x $@

.build/aarch64-linux-musl-gcc:
	@mkdir -p .build
	@printf '#!/bin/sh\nexec zig cc -target aarch64-linux-musl "$$@"\n' > $@
	@chmod +x $@

# ── Go builds (via mage, flags defined in Magefile.go) ────────
build-go-amd64: .build/x86_64-linux-musl-gcc
	PATH="$(CURDIR)/.build:$$PATH" mage build:linux

build-go-arm64: .build/aarch64-linux-musl-gcc
	PATH="$(CURDIR)/.build:$$PATH" mage build:linuxARM64

build-ts:
	npm run build

docker-up:
	docker compose up -d

clean:
	cd rsod && cargo clean
	rm -rf node_modules
	rm -rf dist/
