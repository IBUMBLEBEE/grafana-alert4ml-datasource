//go:build mage
// +build mage

package main

import (
	"fmt"
	"os"

	// mage:import

	"github.com/magefile/mage/mg"
	"github.com/magefile/mage/sh"
)

// Default configures the default target.
var Default = Build.All

// Build contains all build-related targets
type Build mg.Namespace

// All builds versions for all platforms
func (Build) All() {
	mg.Deps(Build.Linux, Build.Darwin, Build.Windows)
}

// Linux builds Linux version
func (Build) Linux() error {
	fmt.Println("Building for Linux...")
	os.Setenv("CGO_ENABLED", "1")
	os.Setenv("GOOS", "linux")
	os.Setenv("GOARCH", "amd64")

	return sh.RunWith(map[string]string{
		"CGO_ENABLED": "1",
		"GOOS":        "linux",
		"GOARCH":      "amd64",
	}, "go", "build", "-o", "dist/gpx_alert4ml_linux_amd64",
		"-tags", "arrow_json_stdlib",
		"-ldflags", "-w -s -X 'github.com/grafana/grafana-plugin-sdk-go/build.buildInfoJSON={\"pluginID\":\"ibumblebee-alert4ml-datasource\",\"version\":\"0.1.0\"}' -X 'main.pluginID=ibumblebee-alert4ml-datasource' -X 'main.version=0.1.0'",
		"./pkg")
}

// Darwin builds macOS ARM64 version
func (Build) Darwin() error {
	fmt.Println("Building for macOS ARM64...")
	return sh.RunWith(map[string]string{
		"CGO_ENABLED": "1",
		"GOOS":        "darwin",
		"GOARCH":      "arm64",
	}, "go", "build", "-o", "dist/gpx_alert4ml_darwin_arm64",
		"-tags", "arrow_json_stdlib",
		"-ldflags", "-w -s -X 'github.com/grafana/grafana-plugin-sdk-go/build.buildInfoJSON={\"pluginID\":\"ibumblebee-alert4ml-datasource\",\"version\":\"0.1.0\"}' -X 'main.pluginID=ibumblebee-alert4ml-datasource' -X 'main.version=0.1.0'",
		"./pkg")
}

// Windows 构建 Windows 版本
func (Build) Windows() error {
	fmt.Println("Building for Windows...")
	return sh.RunWith(map[string]string{
		"CGO_ENABLED": "1",
		"GOOS":        "windows",
		"GOARCH":      "amd64",
	}, "go", "build", "-o", "dist/gpx_alert4ml_windows_amd64.exe",
		"-tags", "arrow_json_stdlib",
		"-ldflags", "-w -s -X 'github.com/grafana/grafana-plugin-sdk-go/build.buildInfoJSON={\"pluginID\":\"ibumblebee-alert4ml-datasource\",\"version\":\"0.1.0\"}' -X 'main.pluginID=ibumblebee-alert4ml-datasource' -X 'main.version=0.1.0'",
		"./pkg")
}
