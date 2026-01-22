.PHONY: all install-frontend install-go-deps build-rs build-go build-ts copy-so docker-up clean

all: pre-build install-frontend install-go-deps build-rs build-go build-ts copy-so docker-up

pre-build:
	go install github.com/bufbuild/buf/cmd/buf@v1.64.0
	go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
	go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

install-frontend:
	npm install

generate-proto:
	buf generate

install-go-deps:
	go mod download

build-rs:
	cd pkg/rsod && cross build --release --target x86_64-unknown-linux-gnu

build-go:
	mage -v build:linux

build-ts:
	npm run build

copy-so:
	mkdir -p dist/
	cp -f pkg/rsod/target/x86_64-unknown-linux-gnu/release/lib*.so dist/

docker-up:
	docker compose up -d

clean:
	cd pkg/rsod && cargo clean
	rm -rf node_modules
	rm -rf dist/
