package plugin

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/google/uuid"
	"github.com/grafana/grafana-plugin-sdk-go/backend/log"
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

// GetPluginDir gets the plugin directory
// Get plugin directory through executable file path
func GetPluginDir() (string, error) {
	// Get executable file path
	execPath, err := os.Executable()
	if err != nil {
		return "", fmt.Errorf("failed to get executable path: %w", err)
	}

	// Resolve symbolic links to get real path
	realPath, err := filepath.EvalSymlinks(execPath)
	if err != nil {
		// If resolution fails, use original path
		realPath = execPath
	}

	// Get executable file directory (usually the plugin directory)
	pluginDir := filepath.Dir(realPath)

	// Log debug information
	log.DefaultLogger.Debug("Plugin directory detected", "path", pluginDir, "executable", execPath)

	return pluginDir, nil
}

// GetPluginDirOrFallback gets plugin directory, uses current working directory as fallback if failed
func GetPluginDirOrFallback() string {
	pluginDir, err := GetPluginDir()
	if err != nil {
		// If failed, fallback to current working directory
		wd, err := os.Getwd()
		if err != nil {
			log.DefaultLogger.Warn("Failed to get plugin directory and working directory, using current directory", "error", err)
			return "." // Final fallback
		}
		log.DefaultLogger.Warn("Failed to get plugin directory, using working directory", "fallback", wd, "error", err)
		return wd
	}
	return pluginDir
}

// GetStoragePath gets storage path (for SQLite database, etc.)
// Create data subdirectory under plugin directory for data storage
func GetStoragePath(filename string) (string, error) {
	pluginDir, err := GetPluginDir()
	if err != nil {
		return "", fmt.Errorf("failed to get plugin directory: %w", err)
	}

	// Create data subdirectory under plugin directory
	dataDir := filepath.Join(pluginDir, "data")

	// Ensure directory exists
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create data directory: %w", err)
	}

	// Return complete file path
	filePath := filepath.Join(dataDir, filename)
	return filePath, nil
}
