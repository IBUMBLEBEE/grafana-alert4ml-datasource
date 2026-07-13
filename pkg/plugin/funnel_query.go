package plugin

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/IBUMBLEBEE/grafana-alert4ml-datasource/pkg/rsod"
	"github.com/IBUMBLEBEE/grafana-alert4ml-datasource/pkg/sdk"
	"github.com/grafana/grafana-plugin-sdk-go/backend"
	"github.com/grafana/grafana-plugin-sdk-go/data"
	"github.com/tidwall/sjson"
)

const defaultFunnelMaxDataPoints int64 = 1500

// effectiveFunnelHistoryInterval coarsens the Grafana panel interval ($__interval) when the
// history window would exceed maxDataPoints — same auto-step idea as $__interval itself.
func effectiveFunnelHistoryInterval(panelIntervalMs int64, durationMs uint64, maxDataPoints int64) int64 {
	if maxDataPoints <= 0 {
		maxDataPoints = defaultFunnelMaxDataPoints
	}
	if panelIntervalMs <= 0 || durationMs == 0 {
		return panelIntervalMs
	}
	coarseMs := int64(durationMs) / maxDataPoints
	if coarseMs <= panelIntervalMs {
		return panelIntervalMs
	}
	return coarseMs
}

func buildTargetsWithInterval(queryJson *Alert4MLQueryJson, refID string, intervalMs int64) ([]json.RawMessage, error) {
	if intervalMs <= 0 {
		return nil, fmt.Errorf("intervalMs must be > 0, got %d", intervalMs)
	}
	out := make([]json.RawMessage, 0, len(queryJson.Targets))
	for _, target := range queryJson.Targets {
		queryStr, err := sjson.Set(string(target), "intervalMs", intervalMs)
		if err != nil {
			return nil, err
		}
		queryStr, err = sjson.Set(queryStr, "refId", refID)
		if err != nil {
			return nil, err
		}
		out = append(out, json.RawMessage(queryStr))
	}
	return out, nil
}

// BuildFunnelDualQueryBodies returns separate upstream queries for profile history and panel current.
//
//	current: [panelFrom, panelTo]               @ panelIntervalMs ($__interval)
//	history: [panelFrom - duration, panelFrom) @ auto-coarsened panel interval
func BuildFunnelDualQueryBodies(
	query backend.DataQuery,
	queryJson *Alert4MLQueryJson,
	htr HistoryTimeRange,
) (*Alert4MLQueryBody, *Alert4MLQueryBody, error) {
	panelFrom := query.TimeRange.From
	panelTo := query.TimeRange.To
	if panelTo.Before(panelFrom) {
		return nil, nil, fmt.Errorf("funnel: panel time range is empty")
	}

	panelInterval := query.Interval.Milliseconds()
	if panelInterval <= 0 {
		return nil, nil, fmt.Errorf("funnel: panel interval must be > 0")
	}
	historyInterval := effectiveFunnelHistoryInterval(panelInterval, htr.DurationMs, query.MaxDataPoints)

	histTargets, err := buildTargetsWithInterval(queryJson, query.RefID, historyInterval)
	if err != nil {
		return nil, nil, err
	}
	curTargets, err := buildTargetsWithInterval(queryJson, query.RefID, panelInterval)
	if err != nil {
		return nil, nil, err
	}

	historyFrom := panelFrom.Add(-time.Duration(htr.DurationMs) * time.Millisecond)
	historyBody := &Alert4MLQueryBody{
		Queries:    histTargets,
		From:       historyFrom,
		To:         panelFrom,
		IntervalMs: historyInterval,
	}
	currentBody := &Alert4MLQueryBody{
		Queries:    curTargets,
		From:       panelFrom,
		To:         panelTo,
		IntervalMs: panelInterval,
	}
	return historyBody, currentBody, nil
}

func frameSeriesKey(f *data.Frame) string {
	if f == nil {
		return ""
	}
	if f.Name != "" {
		return f.Name
	}
	if len(f.Fields) > 0 {
		return f.Fields[0].Labels.String()
	}
	return ""
}

func matchHistoryFrame(historyFrames []*data.Frame, current *data.Frame, frameIdx int) *data.Frame {
	if frameIdx >= 0 && frameIdx < len(historyFrames) && historyFrames[frameIdx] != nil {
		return historyFrames[frameIdx]
	}
	if current == nil {
		return nil
	}
	key := frameSeriesKey(current)
	for _, h := range historyFrames {
		if h != nil && frameSeriesKey(h) == key {
			return h
		}
	}
	return nil
}

func cloneFrameFromResponse(resp backend.DataResponse, frameIdx int, match *data.Frame) *data.Frame {
	cp := resp.DeepCopy()
	if frameIdx >= 0 && frameIdx < len(cp.Frames) && cp.Frames[frameIdx] != nil {
		return cp.Frames[frameIdx]
	}
	if match != nil {
		for _, fr := range cp.Frames {
			if fr != nil && fr.Name == match.Name {
				return fr
			}
		}
	}
	return nil
}

func processFunnelDualQuery(
	client *sdk.GrafanaClient,
	query backend.DataQuery,
	queryJson *Alert4MLQueryJson,
	hyperParams HyperParams,
) ([]*data.Frame, *backend.DataResponse, error) {
	htr := effectiveHistoryTimeRange(queryJson.DetectType, queryJson.HistoryTimeRange)
	historyBody, currentBody, err := BuildFunnelDualQueryBodies(query, queryJson, htr)
	if err != nil {
		return nil, nil, err
	}

	histRsp, err := client.DataSourceQuery(historyBody)
	if err != nil {
		return nil, nil, fmt.Errorf("funnel history query: %w", err)
	}
	curRsp, err := client.DataSourceQuery(currentBody)
	if err != nil {
		return nil, nil, fmt.Errorf("funnel current query: %w", err)
	}

	histResponse, ok := histRsp.Responses[query.RefID]
	if !ok || len(histResponse.Frames) == 0 {
		return nil, nil, fmt.Errorf("funnel history query returned no frames for refId %q", query.RefID)
	}
	curResponse, ok := curRsp.Responses[query.RefID]
	if !ok || len(curResponse.Frames) == 0 {
		return nil, nil, fmt.Errorf("funnel current query returned no frames for refId %q", query.RefID)
	}

	fp := hyperParams.(*FunnelHyperParams)
	periods, err := ParsePeriods(fp.Periods, currentBody.IntervalMs)
	if err != nil {
		return nil, nil, err
	}

	persistProfile := true
	if fp.PersistProfile != nil {
		persistProfile = *fp.PersistProfile
	}

	options := rsod.FunnelOptions{
		UUID:                 "", // filled per frame below
		Trend:                funnelTrendForRust(fp.Trend),
		BucketSlotSecs:       fp.BucketSlotSecs,
		AutoTrend:            fp.AutoTrend,
		KOuter:               fp.KOuter,
		KInner:               fp.KInner,
		MinSamples:           fp.MinSamples,
		StdDevMultiplier:     fp.StdDevMultiplier,
		EnableL2:             false,
		PersistProfile:       persistProfile,
		Periods:              periods,
		ModelName:            fp.ModelName,
		MaxSparseBucketRatio: fp.MaxSparseBucketRatio,
		LookbackDays:         fp.LookbackDays,
		EvalWindowSecs:       fp.EvalWindowSecs,
		AlertOutputMode:      fp.AlertOutputMode,
	}

	newframes := make([]*data.Frame, 0)
	for frameIdx, f := range curResponse.Frames {
		if f == nil || len(f.Fields) == 0 {
			continue
		}

		currentFrame := cloneFrameFromResponse(curResponse, frameIdx, f)
		if currentFrame == nil {
			continue
		}

		historyFrame := matchHistoryFrame(histResponse.Frames, f, frameIdx)
		historyCopy := cloneFrameFromResponse(histResponse, frameIdx, f)
		if historyCopy == nil && historyFrame != nil {
			historyCopy = cloneFrameFromResponse(histResponse, frameIdx, historyFrame)
		}
		if historyCopy == nil {
			historyCopy = data.NewFrame(f.Name)
		}

		historyCopy, currentFrame, err = ensureFunnelFrames(nil, historyCopy, currentFrame)
		if err != nil {
			return nil, nil, err
		}
		if err = TransformDataFrame(currentFrame); err != nil {
			return nil, nil, err
		}
		if err = TransformDataFrame(historyCopy); err != nil {
			return nil, nil, err
		}

		uk := UniqueKeysUUID{
			DetectType:    queryJson.DetectType,
			SupportDetect: queryJson.SupportDetect,
			UniqueKeys:    queryJson.UniqueKeys,
			SeriesName:    f.Name,
		}
		ukUUID, err := uk.ToUUIDString()
		if err != nil {
			return nil, nil, err
		}
		options.UUID = ukUUID

		resultFunnelDF, err := rsod.FunnelFitPredict(currentFrame, historyCopy, options)
		if err != nil {
			return nil, nil, err
		}

		newframe := RenderFrameWithBaseline(resultFunnelDF, query.RefID)
		if queryJson.ShowAnomalyPoints {
			removeNonAnomalyFields(newframe)
		}
		newframes = append(newframes, newframe)
	}

	return newframes, &curResponse, nil
}
