# Multi-stage: musl binary for alert4ml mock-metrics.

FROM docker.m.daocloud.io/library/rust:1.85-alpine AS builder
RUN apk add --no-cache musl-dev
WORKDIR /app
COPY .cargo ./.cargo
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

FROM docker.m.daocloud.io/library/alpine:3.21
COPY --from=builder /app/target/release/mock-metrics /usr/local/bin/mock-metrics
EXPOSE 9108
CMD ["mock-metrics", "--host", "0.0.0.0", "--port", "9108"]
