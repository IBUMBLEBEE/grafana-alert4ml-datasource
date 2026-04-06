//go:build mage
// +build mage

package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/magefile/mage/mg"
	"github.com/magefile/mage/sh"
)

const pluginID = "ibumblebee-alert4ml-datasource"
const pluginBinary = "gpx_alert4ml"

var (
	rustTargetAMD64 = "x86_64-unknown-linux-musl"
	rustTargetARM64 = "aarch64-unknown-linux-musl"
)

// ── Build namespace ──────────────────────────────────────────

type Build mg.Namespace

// packageJSON is the minimal structure we need from package.json.
type packageJSON struct {
	Version string `json:"version"`
}

func readVersion() (string, error) {
	data, err := os.ReadFile("package.json")
	if err != nil {
		return "", fmt.Errorf("read package.json: %w", err)
	}
	var p packageJSON
	if err := json.Unmarshal(data, &p); err != nil {
		return "", fmt.Errorf("parse package.json: %w", err)
	}
	return p.Version, nil
}

func buildGo(goos, goarch, cc, rustTarget string) error {
	if err := os.MkdirAll("dist", 0o755); err != nil {
		return err
	}

	root, err := os.Getwd()
	if err != nil {
		return err
	}

	env := map[string]string{
		"CGO_ENABLED": "1",
		"CC":          cc,
		"GOOS":        goos,
		"GOARCH":      goarch,
		"CGO_LDFLAGS": fmt.Sprintf("-L%s", filepath.Join(root, "rsod", "target", rustTarget, "release")),
	}

	// Read version from package.json
	version, err := readVersion()
	if err != nil {
		return err
	}

	// Build info JSON matches github.com/grafana/grafana-plugin-sdk-go/build/buildinfo.Info
	buildInfo, _ := json.Marshal(map[string]interface{}{
		"time":     time.Now().Unix(),
		"pluginID": pluginID,
		"version":  version,
	})

	ldflags := fmt.Sprintf(
		"-linkmode external -extldflags '-static' -X 'github.com/grafana/grafana-plugin-sdk-go/build/buildinfo.buildInfoJSON=%s'",
		string(buildInfo),
	)

	output := filepath.Join("dist", fmt.Sprintf("%s_%s_%s", pluginBinary, goos, goarch))

	return sh.RunWithV(env, "go", "build",
		"-ldflags", ldflags,
		"-o", output,
		"./pkg")
}

// Linux builds Go plugin for linux/amd64 (static, no glibc).
func (Build) Linux() error {
	return buildGo("linux", "amd64", "x86_64-linux-musl-gcc", rustTargetAMD64)
}

// LinuxARM64 builds Go plugin for linux/arm64 (static, no glibc).
func (Build) LinuxARM64() error {
	return buildGo("linux", "arm64", "aarch64-linux-musl-gcc", rustTargetARM64)
}

// RsAMD64 builds Rust library for linux/amd64 (musl static) using cargo-zigbuild.
func (Build) RsAMD64() error {
	return sh.RunV("cargo", "zigbuild", "--release",
		"--target", rustTargetAMD64,
		"--manifest-path", filepath.Join("rsod", "Cargo.toml"))
}

// RsARM64 builds Rust library for linux/arm64 (musl static) using cargo-zigbuild.
func (Build) RsARM64() error {
	return sh.RunV("cargo", "zigbuild", "--release",
		"--target", rustTargetARM64,
		"--manifest-path", filepath.Join("rsod", "Cargo.toml"))
}

// TS builds the frontend.
func (Build) TS() error {
	return sh.RunV("npm", "run", "build")
}

// All builds Rust + Go (amd64) + frontend.
func (Build) All() error {
	b := Build{}
	if err := b.RsAMD64(); err != nil {
		return err
	}
	if err := b.Linux(); err != nil {
		return err
	}
	return b.TS()
}

// AllPlatforms builds Rust + Go for both amd64 and arm64, plus frontend.
func (Build) AllPlatforms() error {
	b := Build{}
	if err := b.RsAMD64(); err != nil {
		return err
	}
	if err := b.RsARM64(); err != nil {
		return err
	}
	if err := b.Linux(); err != nil {
		return err
	}
	if err := b.LinuxARM64(); err != nil {
		return err
	}
	return b.TS()
}

// ── Top-level targets ────────────────────────────────────────

// Clean removes build artifacts.
func Clean() error {
	_ = sh.RunV("cargo", "clean", "--manifest-path", filepath.Join("rsod", "Cargo.toml"))
	return sh.Rm("dist")
}

// Default target.
var Default = Build.All
