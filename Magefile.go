//go:build mage
// +build mage

package main

import (
	// mage:import
	build "github.com/grafana/grafana-plugin-sdk-go/build"
)

var Default = build.SetBeforeBuildCallback(func(cfg build.Config) (build.Config, error) {
	// 修改配置
	cfg.EnableCGo = true
	return cfg, nil
})
