.PHONY: all all-platforms install-frontend build-backend-amd64 build-backend-arm64 build-ts reload docker-up clean

# ── Backend (Rust plugin, musl static) ───────────────────────
RUST_TARGET_AMD64 = x86_64-unknown-linux-musl
RUST_TARGET_ARM64 = aarch64-unknown-linux-musl
BACKEND_CRATE = rsod-backend
PLUGIN_BINARY = gpx_alert4ml

# ── Default: build amd64 ─────────────────────────────────────
all: install-frontend build-backend-amd64 build-backend-arm64 build-ts docker-up clean

# ── All platforms ────────────────────────────────────────────
all-platforms: build-backend-amd64 build-backend-arm64 build-ts

install-frontend:
	npm install

# ── Backend builds ───────────────────────────────────────────
# Build the plugin backend crate and copy the binary into dist/
# using the Grafana-executable naming (gpx_alert4ml_linux_<arch>).
# Copy via tmp+mv: dist/ may be bind-mounted by a running Grafana that is
# executing the binary; a direct cp overwrite triggers ETXTBSY (text file busy).
# rename atomically replaces the directory entry and is unaffected.
build-backend-amd64:
	cd rsod && cargo zigbuild --release --target $(RUST_TARGET_AMD64) -p $(BACKEND_CRATE)
	mkdir -p dist
	cp rsod/target/$(RUST_TARGET_AMD64)/release/$(PLUGIN_BINARY) dist/$(PLUGIN_BINARY)_linux_amd64.tmp
	mv -f dist/$(PLUGIN_BINARY)_linux_amd64.tmp dist/$(PLUGIN_BINARY)_linux_amd64

build-backend-arm64:
	cd rsod && cargo zigbuild --release --target $(RUST_TARGET_ARM64) -p $(BACKEND_CRATE)
	mkdir -p dist
	cp rsod/target/$(RUST_TARGET_ARM64)/release/$(PLUGIN_BINARY) dist/$(PLUGIN_BINARY)_linux_arm64.tmp
	mv -f dist/$(PLUGIN_BINARY)_linux_arm64.tmp dist/$(PLUGIN_BINARY)_linux_arm64

build-ts:
	npm run build

# ── Local test loop ──────────────────────────────────────────
# One-shot effect after changing Rust backend code: recompile (incremental) → restart container.
# Grafana does not auto-restart the backend plugin process; the container must be
# restarted for the new binary to take effect.
reload: build-backend-amd64
	docker compose restart

docker-up:
	docker compose up -d

clean:
	cd rsod && cargo clean
	rm -rf node_modules
	rm -rf dist/
