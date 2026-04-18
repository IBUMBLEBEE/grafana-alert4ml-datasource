package plugin

import (
	"encoding/json"
	"fmt"

	"github.com/google/uuid"
)

// Define namespace UUID for generating deterministic UUID v5
// Can use uuid.NameSpaceOID or custom namespace
var (
	// Alert4MLNamespace is the namespace UUID for Alert4ML plugin
	// Generated using uuid.NewSHA1(uuid.NameSpaceOID, []byte("alert4ml-plugin"))
	Alert4MLNamespace = uuid.MustParse("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
)

type UniqueKeysUUID struct {
	DetectType    string     `json:"detectType"`
	SupportDetect string     `json:"supportDetect"`
	UniqueKeys    UniqueKeys `json:"uniqueKeys"`
	SeriesName    string     `json:"seriesName"`
}

// ToUUID converts UniqueKeysUUID to UUID v5
// Uses SHA-1 hash to ensure same input always generates same UUID
func (uk *UniqueKeysUUID) ToUUID() (uuid.UUID, error) {
	// Serialize struct to JSON to ensure consistent field order
	jsonData, err := json.Marshal(uk)
	if err != nil {
		return uuid.Nil, fmt.Errorf("failed to marshal UniqueKeysUUID: %w", err)
	}

	// Use UUID v5 (namespace-based deterministic UUID)
	// NewSHA1 generates UUID using SHA-1 hash algorithm
	result := uuid.NewSHA1(Alert4MLNamespace, jsonData)
	return result, nil
}

// ToUUIDString converts UniqueKeysUUID to UUID string
func (uk *UniqueKeysUUID) ToUUIDString() (string, error) {
	u, err := uk.ToUUID()
	if err != nil {
		return "", err
	}
	return u.String(), nil
}

// DeriveUUID creates a new deterministic UUID by combining a base UUID with
// additional data (e.g. training parameters). Same inputs always produce the
// same derived UUID, different inputs produce different UUIDs.
func DeriveUUID(baseUUID string, extra interface{}) (string, error) {
	base, err := uuid.Parse(baseUUID)
	if err != nil {
		return "", fmt.Errorf("failed to parse base UUID: %w", err)
	}
	extraJSON, err := json.Marshal(extra)
	if err != nil {
		return "", fmt.Errorf("failed to marshal extra data: %w", err)
	}
	derived := uuid.NewSHA1(base, extraJSON)
	return derived.String(), nil
}
