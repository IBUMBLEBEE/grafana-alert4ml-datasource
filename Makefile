.PHONY: all install-frontend install-go-deps build-rs build-go build-ts copy-so docker-up clean

# Compilation targets
TARGETS ?= x86_64-unknown-linux-gnu

all: install-frontend install-go-deps build-rs build-go build-ts copy-so docker-up

install-frontend:
	npm install

install-go-deps:
	go mod download

build-rs:
	cd pkg/rsod && cross build --release --target $(TARGETS)

build-go:
	mage -v build:linux

build-ts:
	npm run build

copy-so:
	mkdir -p dist/
	cp -f rsod/target/$(TARGETS)/release/lib*.so dist/

docker-up:
	docker compose up -d

clean:
	cd pkg/rsod && cargo clean
	rm -rf node_modules
	rm -rf dist/
