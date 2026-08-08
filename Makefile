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
# 拷贝使用 tmp+mv：dist/ 可能被运行中的 Grafana bind-mount 并正在执行，
# 直接 cp 覆盖会触发 ETXTBSY（text file busy）；rename 原子替换目录项不受影响。
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

# ── 本地测试循环 ─────────────────────────────────────────────
# 改了 Rust 后端代码后一键生效：重编译（增量）→ 重启容器。
# Grafana 不会自动重启后端插件进程，必须重启容器让新二进制生效。
reload: build-backend-amd64
	docker compose restart

docker-up:
	docker compose up -d

clean:
	cd rsod && cargo clean
	rm -rf node_modules
	rm -rf dist/
