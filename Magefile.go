//go:build mage
// +build mage

package main

import (
	// mage:import
	"log"

	build "github.com/grafana/grafana-plugin-sdk-go/build"
	"github.com/magefile/mage/mg"
	"github.com/magefile/mage/sh"
)

func BuildAll() error {
	build.SetBeforeBuildCallback(func(cfg build.Config) (build.Config, error) {
		err := sh.Copy("dist/gpx_alert4ml_rsod_server", "rust_dist/gpx_alert4ml_rsod_server")
		if err != nil {
			log.Fatalf("Failed to copy gpx_alert4ml_rsod_server: %v", err)
		}
		return cfg, nil
	})
	mg.Deps(build.BuildAll)
	return nil
}

var Default = BuildAll
