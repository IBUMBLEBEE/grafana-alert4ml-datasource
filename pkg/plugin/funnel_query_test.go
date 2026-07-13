package plugin

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/grafana/grafana-plugin-sdk-go/backend"
	"github.com/grafana/grafana-plugin-sdk-go/data"
)

func TestBuildFunnelDualQueryBodies(t *testing.T) {
	panelFrom := time.Date(2026, 6, 10, 12, 0, 0, 0, time.UTC)
	panelTo := panelFrom.Add(6 * time.Hour)
	sevenDays := uint64(7 * 24 * 60 * 60 * 1000)

	query := backend.DataQuery{
		RefID: "A",
		TimeRange: backend.TimeRange{
			From: panelFrom,
			To:   panelTo,
		},
		Interval: time.Minute,
	}
	queryJson := &Alert4MLQueryJson{
		Targets: []json.RawMessage{json.RawMessage(`{"refId":"A"}`)},
	}
	htr := HistoryTimeRange{
		DurationMs: sevenDays,
	}

	hist, cur, err := BuildFunnelDualQueryBodies(query, queryJson, htr)
	if err != nil {
		t.Fatalf("BuildFunnelDualQueryBodies: %v", err)
	}

	wantHistFrom := panelFrom.Add(-time.Duration(sevenDays) * time.Millisecond)
	if !hist.From.Equal(wantHistFrom) {
		t.Errorf("history From = %v, want %v", hist.From, wantHistFrom)
	}
	if !hist.To.Equal(panelFrom) {
		t.Errorf("history To = %v, want panelFrom %v", hist.To, panelFrom)
	}
	wantHistInterval := effectiveFunnelHistoryInterval(
		int64(time.Minute/time.Millisecond),
		sevenDays,
		0,
	)
	if hist.IntervalMs != wantHistInterval {
		t.Errorf("history IntervalMs = %d, want coarsened %d", hist.IntervalMs, wantHistInterval)
	}

	if !cur.From.Equal(panelFrom) {
		t.Errorf("current From = %v, want %v", cur.From, panelFrom)
	}
	if !cur.To.Equal(panelTo) {
		t.Errorf("current To = %v, want %v", cur.To, panelTo)
	}
	if cur.IntervalMs != int64(time.Minute/time.Millisecond) {
		t.Errorf("current IntervalMs = %d, want 60000", cur.IntervalMs)
	}
}

func TestEffectiveFunnelHistoryInterval(t *testing.T) {
	panel := int64(60_000) // 1m
	sevenDays := uint64(7 * 24 * 60 * 60 * 1000)

	got := effectiveFunnelHistoryInterval(panel, sevenDays, 1500)
	if got <= panel {
		t.Fatalf("expected history interval coarser than panel, got %d", got)
	}

	short := effectiveFunnelHistoryInterval(panel, 300_000, 1500)
	if short != panel {
		t.Fatalf("short history should keep panel interval, got %d", short)
	}
}

func TestMatchHistoryFramePrefersIndex(t *testing.T) {
	cur0 := data.NewFrame("")
	cur0.Fields = append(cur0.Fields, data.NewField("value", data.Labels{"instance": "a"}, []float64{1}))
	cur1 := data.NewFrame("")
	cur1.Fields = append(cur1.Fields, data.NewField("value", data.Labels{"instance": "b"}, []float64{2}))

	hist0 := data.NewFrame("up")
	hist0.Fields = append(hist0.Fields, data.NewField("value", data.Labels{"instance": "a"}, []float64{10}))
	hist1 := data.NewFrame("up")
	hist1.Fields = append(hist1.Fields, data.NewField("value", data.Labels{"instance": "b"}, []float64{20}))

	frames := []*data.Frame{hist0, hist1}
	if got := matchHistoryFrame(frames, cur1, 1); got != hist1 {
		t.Fatalf("expected hist1 by index, got %v", got)
	}
}
