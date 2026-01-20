package sdk

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/grafana/grafana-plugin-sdk-go/backend"
)

type GrafanaClient struct {
	BaseURL    string
	APIKey     string
	HTTPClient *http.Client
}

func NewGrafanaClient(baseURL, apiKey string) *GrafanaClient {
	return &GrafanaClient{
		BaseURL: baseURL,
		APIKey:  apiKey,
		HTTPClient: &http.Client{
			Timeout: 2 * time.Minute,
		},
	}
}

func (c *GrafanaClient) doRequest(method, endpoint string, body any) ([]byte, error) {
	url := fmt.Sprintf("%s%s", c.BaseURL, endpoint)

	var reqBody []byte
	var err error
	if body != nil {
		reqBody, err = json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request body: %w", err)
		}
	}

	req, err := http.NewRequest(method, url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+c.APIKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("received error response: %s", resp.Status)
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	return respBody, nil
}

func (c *GrafanaClient) GetDashboard(uid string) (map[string]json.RawMessage, error) {
	endpoint := fmt.Sprintf("/api/dashboards/uid/%s", uid)
	respBody, err := c.doRequest(http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}

	var dashboard map[string]json.RawMessage
	err = json.Unmarshal(respBody, &dashboard)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return dashboard, nil
}

func (c *GrafanaClient) DataSourceQuery(query any) (*backend.QueryDataResponse, error) {
	endpoint := "/api/ds/query"
	respBody, err := c.doRequest(http.MethodPost, endpoint, query)
	if err != nil {
		return nil, err
	}

	var result backend.QueryDataResponse
	err = json.Unmarshal(respBody, &result)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	// refAresp := result.Responses["A"]
	// for k, v := range result.Responses {
	// 	if k == "A" {
	// 		continue
	// 	}
	// 	refAresp.Frames = append(refAresp.Frames, v.Frames...)
	// }
	// bq := backend.QueryDataResponse{
	// 	Responses: map[string]backend.DataResponse{
	// 		"A": refAresp,
	// 	},
	// }
	// // result.Responses["A"] = refAresp
	return &result, nil
}

func (c *GrafanaClient) LoginPing() error {
	endpoint := "/api/login/ping"
	_, err := c.doRequest(http.MethodGet, endpoint, nil)
	if err != nil {
		return err
	}
	return nil
}
