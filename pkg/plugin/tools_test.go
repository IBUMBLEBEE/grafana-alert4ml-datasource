package plugin

import (
	"fmt"
	"testing"
	"time"

	"github.com/grafana/grafana-plugin-sdk-go/data"
)

func TestParsePeriods(t *testing.T) {
	periods, err := ParsePeriods("1h,2h,3h", 3600000)
	if err != nil {
		t.Fatalf("ParsePeriods failed: %v", err)
	}
	if len(periods) != 3 {
		t.Fatalf("expected 3 periods, got %v", periods)
	}
	t.Logf("periods: %v", periods)
}

func TestParsePeriodsBareIntegerAsHours(t *testing.T) {
	periods, err := ParsePeriods("24", 3600000)
	if err != nil {
		t.Fatalf("ParsePeriods(24): %v", err)
	}
	if len(periods) != 1 || periods[0] != 24 {
		t.Fatalf("expected [24], got %v", periods)
	}
}

func TestParsePeriodsEmpty(t *testing.T) {
	periods, err := ParsePeriods("", 3600000)
	if err != nil {
		t.Fatalf("ParsePeriods empty: %v", err)
	}
	if len(periods) != 0 {
		t.Fatalf("expected empty, got %v", periods)
	}
}

func TestParsePeriodsInvalid(t *testing.T) {
	_, err := ParsePeriods("not-a-duration", 3600000)
	if err == nil {
		t.Fatal("expected error for invalid duration")
	}
	fmt.Println(err)
}

func TestEffectiveHistoryTimeRangeFunnelDefault(t *testing.T) {
	htr := effectiveHistoryTimeRange("funnel", HistoryTimeRange{})
	if htr.DurationMs != DefaultFunnelHistoryDurationMs {
		t.Fatalf("expected funnel default 7d, got %d", htr.DurationMs)
	}
}

func TestEffectiveHistoryTimeRangePreservesExplicit(t *testing.T) {
	explicit := HistoryTimeRange{DurationMs: 300_000}
	htr := effectiveHistoryTimeRange("funnel", explicit)
	if htr.DurationMs != 300_000 {
		t.Fatalf("expected explicit duration preserved, got %d", htr.DurationMs)
	}
}

func TestEnsureFunnelFramesColdStartFromCurrent(t *testing.T) {
	ts := time.Date(2025, 5, 10, 0, 0, 0, 0, time.UTC)
	current := data.NewFrame("s")
	current.Fields = append(current.Fields,
		data.NewField("time", nil, make([]time.Time, 30)),
		data.NewField("value", nil, make([]float64, 30)),
	)
	for i := 0; i < 30; i++ {
		current.Fields[0].Set(i, ts.Add(time.Duration(i)*time.Hour))
		current.Fields[1].Set(i, float64(i))
	}
	emptyHistory := data.NewFrame("s")
	emptyHistory.Fields = append(emptyHistory.Fields,
		data.NewField("time", nil, []time.Time{}),
		data.NewField("value", nil, []float64{}),
	)

	h, c, err := ensureFunnelFrames(nil, emptyHistory, current)
	if err != nil {
		t.Fatalf("ensureFunnelFrames: %v", err)
	}
	if h.Fields[0].Len() == 0 || c.Fields[0].Len() == 0 {
		t.Fatalf("expected non-empty split, history=%d current=%d", h.Fields[0].Len(), c.Fields[0].Len())
	}
}

func TestEnsureFunnelFramesEvalFromHistoryTail(t *testing.T) {
	ts := time.Date(2025, 5, 1, 0, 0, 0, 0, time.UTC)
	history := data.NewFrame("s")
	history.Fields = append(history.Fields,
		data.NewField("time", nil, make([]time.Time, 30)),
		data.NewField("value", nil, make([]float64, 30)),
	)
	for i := 0; i < 30; i++ {
		history.Fields[0].Set(i, ts.Add(time.Duration(i)*time.Hour))
		history.Fields[1].Set(i, float64(i))
	}
	emptyCurrent := data.NewFrame("s")
	emptyCurrent.Fields = append(emptyCurrent.Fields,
		data.NewField("time", nil, []time.Time{}),
		data.NewField("value", nil, []float64{}),
	)

	h, c, err := ensureFunnelFrames(nil, history, emptyCurrent)
	if err != nil {
		t.Fatalf("ensureFunnelFrames: %v", err)
	}
	if h.Fields[0].Len() == 0 || c.Fields[0].Len() == 0 {
		t.Fatalf("expected non-empty split, history=%d current=%d", h.Fields[0].Len(), c.Fields[0].Len())
	}
}
