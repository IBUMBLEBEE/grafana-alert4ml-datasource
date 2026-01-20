package plugin

import (
	"fmt"
	"testing"
)

func TestParsePeriods(t *testing.T) {
	periods, err := ParsePeriods("1h,2h,3h", 3600000)
	if err != nil {
		t.Fatalf("ParsePeriods failed: %v", err)
	}
	fmt.Println(periods)
	t.Logf("periods: %v", periods)
}
