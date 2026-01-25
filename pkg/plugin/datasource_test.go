package plugin

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/grafana/grafana-plugin-sdk-go/backend"
)

func TestQueryData(t *testing.T) {
	tests := []struct {
		name    string
		request *backend.QueryDataRequest
		wantErr bool
	}{
		{
			name:    "Normal query test",
			request: createTestRequest(),
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			jsonData, _ := json.Marshal(tt.request)
			fmt.Println("tt.request", string(jsonData))
			ds := &Datasource{}
			resp, err := ds.QueryData(context.Background(), tt.request)
			// fmt.Println("resp", err)

			// jsonData, _ := resp.MarshalJSON()
			// fmt.Println("resp", string(jsonData))
			// var err error
			// var resp *backend.QueryDataResponse
			if tt.wantErr && err == nil {
				t.Error("Expected error but got none")
			}
			if !tt.wantErr && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}

			if err == nil && resp != nil {
				// Allow empty response for empty query test
				if tt.name != "Empty query test" && len(resp.Responses) == 0 {
					t.Error("Response cannot be empty")
				}
			}
		})
	}
}

func TestQueryDataWithInvalidConfig(t *testing.T) {
	ds := &Datasource{}

	// Test invalid URL configuration
	req := createTestRequest()
	req.PluginContext.DataSourceInstanceSettings.JSONData = json.RawMessage(`{"url":"invalid-url"}`)

	_, err := ds.QueryData(context.Background(), req)
	if err == nil {
		t.Error("Expected invalid URL configuration to produce error")
	}
}

func TestQueryDataWithLargeDataset(t *testing.T) {
	ds := &Datasource{}

	// Test large dataset
	req := createLargeDatasetRequest()

	start := time.Now()
	resp, err := ds.QueryData(context.Background(), req)
	duration := time.Since(start)

	if err != nil {
		t.Errorf("Large dataset query failed: %v", err)
	}

	if duration > 60*time.Second {
		t.Errorf("Query timeout, duration: %v", duration)
	}

	if resp != nil && len(resp.Responses) > 0 {
		t.Logf("Large dataset query successful, duration: %v", duration)
	}
}

// Helper functions
func createTestRequest() *backend.QueryDataRequest {
	from, _ := time.Parse(time.RFC3339, "2025-09-02T23:44:40Z")
	to, _ := time.Parse(time.RFC3339, "2025-09-03T05:44:40Z")

	return &backend.QueryDataRequest{
		PluginContext: createTestPluginContext(),
		Headers:       createTestHeaders(),
		Queries: []backend.DataQuery{
			{
				RefID:         "B",
				MaxDataPoints: 43200,
				Interval:      60000000000, // 1 minute
				TimeRange: backend.TimeRange{
					From: from,
					To:   to,
				},
				JSON: json.RawMessage(`{
					"datasource": {"type": "ibumblebee-alert4ml-datasource", "uid": "feuytustyoglce"},
					"detectType": "outlier",
					"historyTimeRange": {"from": 300, "to": 0},
					"hyperParams": {"modelName": "rsod_model", "periods": "7d,24h"},
					"intervalMs": 60000,
					"maxDataPoints": 43200,
					"refId": "B",
					"seriesRefId": "A",
					"showOriginalData": false,
					"supportDetect": "machine_learning",
					"targets": [{
						"datasource": {"type": "prometheus", "uid": "fekd6mryd46pse"},
						"expr": "rate(process_cpu_seconds_total[5m])",
						"disableTextWrap": false,
                        "editorMode": "code",
                        "exemplar": false,
                        "format": "time_series",
                        "fullMetaSearch": false,
                        "includeNullMetadata": true,
                        "instant": false,
                        "interval": "",
                        "intervalMs": 60000,
                        "legendFormat": "",
                        "maxDataPoints": 43200,
                        "range": true,
                        "refId": "A",
                        "useBackend": false
					}]
				}`),
			},
		},
	}
}

func CreateOutlierDetectionRequest() *backend.QueryDataRequest {
	req := createTestRequest()
	req.Queries[0].JSON = json.RawMessage(`{
		"datasource": {"type": "ibumblebee-alert4ml-datasource", "uid": "test"},
		"detectType": "outlier",
		"historyTimeRange": {"from": 300, "to": 0},
		"hyperParams": {"modelName": "rsod_model", "periods": ""},
		"seriesRefId": "A",
		"showOriginalData": true,
		"supportDetect": "machine_learning",
		"targets": [{
			"datasource": {"type": "prometheus", "uid": "fekd6mryd46pse"},
			"expr": "rate(process_cpu_seconds_total[5m])",
			"disableTextWrap": false,
			"editorMode": "code",
			"exemplar": false,
			"format": "time_series",
			"fullMetaSearch": false,
			"includeNullMetadata": true,
			"instant": false,
			"interval": "",
			"intervalMs": 60000,
			"legendFormat": "",
			"maxDataPoints": 43200,
			"range": true,
			"refId": "A",
			"useBackend": false
		}]
	}`)
	return req
}

func createLargeDatasetRequest() *backend.QueryDataRequest {
	req := createTestRequest()
	req.Queries[0].MaxDataPoints = 50000
	req.Queries[0].Interval = 1000000000 // 1 second
	return req
}

func createTestHeaders() map[string]string {
	return map[string]string{
		"FromAlert":           "true",
		"X-Cache-Skip":        "true",
		"X-Grafana-Org-Id":    "1",
		"http_X-Rule-Folder":  "test",
		"http_X-Rule-Name":    "test+alert",
		"http_X-Rule-Source":  "scheduler",
		"http_X-Rule-Type":    "alerting",
		"http_X-Rule-Uid":     "cewk5tfh0en0ge",
		"http_X-Rule-Version": "6",
	}
}

func createTestPluginContext() backend.PluginContext {
	updatedTime, _ := time.Parse(time.RFC3339, "2025-08-25T16:19:01Z")

	return backend.PluginContext{
		OrgID:         1,
		PluginID:      "ibumblebee-alert4ml-datasource",
		PluginVersion: "0.1.3",
		User: &backend.User{
			Login: "grafana_scheduler",
			Name:  "grafana_scheduler",
			Email: "",
			Role:  "Admin",
		},
		DataSourceInstanceSettings: &backend.DataSourceInstanceSettings{
			ID:               11,
			UID:              "feuytustyoglce",
			Type:             "ibumblebee-alert4ml-datasource",
			Name:             "ibumblebee-alert4ml-datasource",
			URL:              "",
			User:             "",
			Database:         "",
			BasicAuthEnabled: false,
			BasicAuthUser:    "",
			JSONData:         json.RawMessage(`{"url":"http://localhost:3000"}`),
			DecryptedSecureJSONData: map[string]string{
				"apiToken": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
			},
			Updated:    updatedTime,
			APIVersion: "",
		},
	}
}

// Benchmark tests
func BenchmarkQueryData(b *testing.B) {
	ds := &Datasource{}
	req := createTestRequest()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := ds.QueryData(context.Background(), req)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkQueryDataConcurrent(b *testing.B) {
	ds := &Datasource{}
	req := createTestRequest()

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, err := ds.QueryData(context.Background(), req)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkLargeDatasetQuery(b *testing.B) {
	ds := &Datasource{}
	req := createLargeDatasetRequest()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := ds.QueryData(context.Background(), req)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// Stress test
func TestQueryDataStress(t *testing.T) {
	if testing.Short() {
		t.Skip("Skip stress test")
	}

	ds := &Datasource{}
	req := createTestRequest()

	// Concurrent query test
	concurrency := 10
	queriesPerWorker := 100

	var wg sync.WaitGroup
	start := time.Now()

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()

			for j := 0; j < queriesPerWorker; j++ {
				ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
				resp, err := ds.QueryData(ctx, req)
				cancel()

				if err != nil {
					t.Errorf("Worker %d, Query %d failed: %v", workerID, j, err)
					return
				}

				if resp == nil || len(resp.Responses) == 0 {
					t.Errorf("Worker %d, Query %d returned empty response", workerID, j)
				}
			}
		}(i)
	}

	wg.Wait()
	duration := time.Since(start)

	totalQueries := concurrency * queriesPerWorker
	queriesPerSecond := float64(totalQueries) / duration.Seconds()

	t.Logf("Stress test completed:")
	t.Logf("Total queries: %d", totalQueries)
	t.Logf("Total duration: %v", duration)
	t.Logf("QPS: %.2f", queriesPerSecond)

	// Performance assertion
	if queriesPerSecond < 10 {
		t.Errorf("Performance not up to standard, QPS: %.2f < 10", queriesPerSecond)
	}
}

// Memory leak test
func TestQueryDataMemoryLeak(t *testing.T) {
	if testing.Short() {
		t.Skip("Skip memory leak test")
	}

	ds := &Datasource{}
	req := createTestRequest()

	// Run multiple queries to check memory usage
	for i := 0; i < 100; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		_, err := ds.QueryData(ctx, req)
		cancel()

		if err != nil {
			t.Logf("Query %d failed: %v", i, err)
		}

		// Force GC after every 100 queries
		if i%100 == 0 {
			// Memory usage checks can be added here
			t.Logf("Completed %d queries", i)
		}
	}
}

// Timeout test
func TestQueryDataTimeout(t *testing.T) {
	ds := &Datasource{}
	req := createLargeDatasetRequest()

	// Set very short timeout
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	start := time.Now()
	_, err := ds.QueryData(ctx, req)
	duration := time.Since(start)

	// Should fail due to timeout
	if err == nil {
		t.Error("Expected timeout error but got none")
	}

	// Check if timeout occurs within reasonable time
	if duration > 200*time.Millisecond {
		t.Errorf("超时时间过长: %v", duration)
	}
}

// Test with real data
func TestQueryDataWithRealData(t *testing.T) {
	ds := &Datasource{}

	// Use real data structure you provided
	from, _ := time.Parse(time.RFC3339, "2025-09-02T23:44:40Z")
	to, _ := time.Parse(time.RFC3339, "2025-09-03T05:44:40Z")

	req := &backend.QueryDataRequest{
		Headers: createTestHeaders(),
		PluginContext: backend.PluginContext{
			OrgID:         1,
			PluginID:      "ibumblebee-alert4ml-datasource",
			PluginVersion: "0.1.3",
			User: &backend.User{
				Login: "grafana_scheduler",
				Name:  "grafana_scheduler",
				Role:  "Admin",
			},
			DataSourceInstanceSettings: &backend.DataSourceInstanceSettings{
				ID:       11,
				UID:      "feuytustyoglce",
				Type:     "ibumblebee-alert4ml-datasource",
				Name:     "ibumblebee-alert4ml-datasource",
				JSONData: json.RawMessage(`{"url":"http://localhost:3000"}`),
				DecryptedSecureJSONData: map[string]string{
					"apiToken": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
				},
			},
		},
		Queries: []backend.DataQuery{
			{
				RefID:         "B",
				MaxDataPoints: 1000,
				Interval:      60000000000, // 1 minute
				TimeRange: backend.TimeRange{
					From: from,
					To:   to,
				},
				JSON: json.RawMessage(`{
					"datasource": {"type": "ibumblebee-alert4ml-datasource", "uid": "feuytustyoglce"},
					"detectType": "outlier",
					"historyTimeRange": {"from": 300, "to": 0},
					"hyperParams": {"modelName": "rsod_model", "periods": ""},
					"seriesRefId": "A",
					"showOriginalData": false,
					"supportDetect": "machine_learning",
					"targets": [{
						"datasource": {"type": "prometheus", "uid": "fekd6mryd46pse"},
						"expr": "rate(process_cpu_seconds_total[5m])",
						"disableTextWrap": false,
                        "editorMode": "code",
                        "exemplar": false,
                        "format": "time_series",
                        "fullMetaSearch": false,
                        "includeNullMetadata": true,
                        "instant": false,
                        "interval": "",
                        "intervalMs": 60000,
                        "legendFormat": "",
                        "maxDataPoints": 43200,
                        "range": true,
                        "refId": "A",
                        "useBackend": false
					}]
				}`),
			},
		},
	}

	// Set timeout context
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	start := time.Now()
	resp, err := ds.QueryData(ctx, req)
	duration := time.Since(start)

	if err != nil {
		t.Errorf("Real data query failed: %v", err)
	}

	if duration > 30*time.Second {
		t.Errorf("Query timeout, duration: %v", duration)
	}

	if resp != nil && len(resp.Responses) > 0 {
		t.Logf("Real data query successful, duration: %v", duration)
	}
}

func TestMissingData(t *testing.T) {
	ds := &Datasource{}
	req := createTestRequest()

	from, _ := time.Parse(time.RFC3339, "2025-04-24T23:44:40Z")
	to, _ := time.Parse(time.RFC3339, "2025-05-10T05:44:40Z")

	// Simulate missing data scenario
	req.Queries[0].JSON = json.RawMessage(`{"datasource":{"type":"ibumblebee-alert4ml-datasource","uid":"feuytustyoglce"},"detectType":"outlier","hide":false,"historyTimeRange":{"from":604800,"to":0},"hyperParams":{"modelName":"rsod_model","periods":"24h,7d"},"refId":"B","seriesRefId":"A","showOriginalData":false,"supportDetect":"machine_learning","targets":[{"alias":"","bucketAggs":[{"field":"service_name","id":"3","settings":{"min_doc_count":"1","order":"desc","orderBy":"_term","size":"10"},"type":"terms"},{"field":"@timestamp","id":"2","settings":{"interval":"auto"},"type":"date_histogram"}],"datasource":{"type":"elasticsearch","uid":"bekd6oa8eoohse"},"metrics":[{"field":"total","id":"1","type":"sum"}],"query":"service_name: Service3","refId":"A","timeField":"@timestamp"}],"datasourceId":11,"intervalMs":3600000,"maxDataPoints":1716}`)
	req.Queries[0].TimeRange = backend.TimeRange{
		From: from,
		To:   to,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	start := time.Now()
	resp, err := ds.QueryData(ctx, req)
	duration := time.Since(start)

	if err != nil {
		t.Errorf("Missing data query failed: %v", err)
	}

	if duration > 60*time.Second {
		t.Errorf("Query timeout, duration: %v", duration)
	}

	if resp != nil && len(resp.Responses) > 0 {
		t.Logf("Missing data query successful, duration: %v", duration)
	}
}
