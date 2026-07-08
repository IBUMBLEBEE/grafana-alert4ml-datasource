package plugin

import (
	"encoding/json"
	"testing"

	"github.com/IBUMBLEBEE/grafana-alert4ml-datasource/pkg/constant"
)

func TestParseFunnelHyperParams(t *testing.T) {
	raw := json.RawMessage(`{
		"modelName": "funnel",
		"evalWindowSecs": 600,
		"alertOutputMode": "dedupe",
		"lookbackDays": 90,
		"persistProfile": true
	}`)

	hp, err := ParseHyperParams(constant.DetectTypeFunnel, raw)
	if err != nil {
		t.Fatalf("ParseHyperParams: %v", err)
	}
	fp, ok := hp.(*FunnelHyperParams)
	if !ok {
		t.Fatalf("expected *FunnelHyperParams, got %T", hp)
	}
	if fp.EvalWindowSecs != 600 {
		t.Errorf("EvalWindowSecs = %d, want 600", fp.EvalWindowSecs)
	}
	if fp.AlertOutputMode != "dedupe" {
		t.Errorf("AlertOutputMode = %q, want dedupe", fp.AlertOutputMode)
	}
	if fp.PersistProfile == nil || !*fp.PersistProfile {
		t.Errorf("PersistProfile should default to true")
	}
}

func TestFunnelTrendForRust(t *testing.T) {
	if got := funnelTrendForRust("weekly"); got == nil || *got != "Weekly" {
		t.Fatalf("funnelTrendForRust(weekly) = %v, want Weekly", got)
	}
}
