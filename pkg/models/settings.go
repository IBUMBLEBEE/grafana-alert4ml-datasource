package models

import (
	"encoding/json"
	"fmt"

	"github.com/grafana/grafana-plugin-sdk-go/backend"
)

type PluginSettings struct {
	URL        string                `json:"url"`
	TrialMode  bool                  `json:"trialMode"`
	PgHost     string                `json:"pgHost"`
	PgPort     int                   `json:"pgPort"`
	PgDatabase string                `json:"pgDatabase"`
	PgUser     string                `json:"pgUser"`
	PgSSLMode  string                `json:"pgSSLMode"`
	Secrets    *SecretPluginSettings `json:"-"`
}

type SecretPluginSettings struct {
	ApiToken   string `json:"apiToken"`
	PgPassword string `json:"pgPassword"`
}

func LoadPluginSettings(source backend.DataSourceInstanceSettings) (*PluginSettings, error) {
	settings := PluginSettings{}
	err := json.Unmarshal(source.JSONData, &settings)
	if err != nil {
		return nil, fmt.Errorf("could not unmarshal PluginSettings json: %w", err)
	}

	settings.Secrets = loadSecretPluginSettings(source.DecryptedSecureJSONData)

	return &settings, nil
}

func loadSecretPluginSettings(source map[string]string) *SecretPluginSettings {
	return &SecretPluginSettings{
		ApiToken:   source["apiToken"],
		PgPassword: source["pgPassword"],
	}
}

// PgDSN returns a PostgreSQL connection string built from the plugin settings.
func (s *PluginSettings) PgDSN() string {
	port := s.PgPort
	if port == 0 {
		port = 5432
	}
	sslmode := s.PgSSLMode
	if sslmode == "" {
		sslmode = "disable"
	}
	return fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
		s.PgHost, port, s.PgUser, s.Secrets.PgPassword, s.PgDatabase, sslmode)
}
