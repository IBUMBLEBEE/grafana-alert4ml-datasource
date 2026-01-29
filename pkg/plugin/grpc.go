package plugin

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/IBUMBLEBEE/grafana-alert4ml-datasource/pkg/gen/rsod"
	"github.com/grafana/grafana-plugin-sdk-go/backend/log"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// initRSODGrpcClient lazily initializes gRPC client for RSOD service
func initRSODGrpcClient() error {
	// Ensure initialization grpc server
	pluginDir := getGrafanaPluginDir()
	rsodGrpcServerOnce.Do(func() {
		// Check if gRPC server binary exists and is executable
		serverBinaryPath := "gpx_alert4ml_rsod_server"
		serverPath := filepath.Join(pluginDir, serverBinaryPath)

		// Start gRPC server in a goroutine
		go func() {
			log.DefaultLogger.Info("Starting gRPC server", "path", serverPath)
			var cmd *exec.Cmd
			cmd = exec.Command(serverPath)
			if err := cmd.Start(); err != nil {
				log.DefaultLogger.Error("Failed to start gRPC server", "error", err)
				if errors.Is(err, os.ErrNotExist) {
					serverPath = filepath.Join("../../dist", serverBinaryPath)
					cmd = exec.Command(serverPath)
					if err := cmd.Start(); err != nil {
						log.DefaultLogger.Error("Failed to start gRPC server from dist", "error", err)
						os.Exit(1)
					}
				} else {
					os.Exit(1)
				}
			}

			log.DefaultLogger.Info(">>> RUST SERVER STARTED WITH PID: ", cmd.Process.Pid)

			// Wait for the server to finish (this should not happen normally)
			if err := cmd.Wait(); err != nil {
				log.DefaultLogger.Error("gRPC server exited with error", "error", err)
			}
		}()

		// Give the server a moment to start up
		time.Sleep(2 * time.Second)
	})
	rsodGrpcOnce.Do(func() {
		// Connect to Unix Domain Socket
		socketPath := filepath.Join(pluginDir, "rsod.sock")
		conn, err := grpc.NewClient(
			fmt.Sprintf("unix://%s", socketPath),
			grpc.WithTransportCredentials(insecure.NewCredentials()),
		)
		if err != nil {
			rsodGrpcErr = fmt.Errorf("failed to connect to RSOD service: %w", err)
			return
		}

		rsodGrpcClient = rsod.NewRsodServiceClient(conn)
		rsodGrpcErr = nil
	})

	// healthTimeTicker := time.NewTicker(2 * time.Second)

	// go func() {
	// 	for range healthTimeTicker.C {
	// 		grpcReq := &rsod.HealthRequest{}
	// 		resp, err := rsodGrpcClient.Health(context.Background(), grpcReq)
	// 		if err != nil {
	// 			log.DefaultLogger.Error("Failed to send health check request", "error", err)
	// 		} else {
	// 			log.DefaultLogger.Info("Received health check response", "Healthy", resp.Healthy, "Version", resp.Version)
	// 		}
	// 	}
	// }()
	return rsodGrpcErr
}

// stringToTrendType converts string trend type to gRPC enum
func stringToTrendType(trendType string) rsod.TrendType {
	switch trendType {
	case "daily":
		return rsod.TrendType_TREND_TYPE_DAILY
	case "weekly":
		return rsod.TrendType_TREND_TYPE_WEEKLY
	case "monthly":
		return rsod.TrendType_TREND_TYPE_MONTHLY
	default:
		return rsod.TrendType_TREND_TYPE_UNSPECIFIED
	}
}
