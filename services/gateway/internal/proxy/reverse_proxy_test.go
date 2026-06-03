package proxy

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/labstack/echo/v4"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestProxyWebSocket(t *testing.T) {
	// 1. Setup upstream WebSocket server
	upgrader := websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool { return true },
	}

	var upstreamReceivedHeaders http.Header
	var upstreamErr error

	upstreamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamReceivedHeaders = r.Header
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			upstreamErr = err
			return
		}
		defer conn.Close()

		for {
			mt, message, err := conn.ReadMessage()
			if err != nil {
				break
			}
			err = conn.WriteMessage(mt, message)
			if err != nil {
				break
			}
		}
	}))
	defer upstreamServer.Close()

	// 2. Setup gateway with Echo and ReverseProxy
	e := echo.New()
	p := NewReverseProxy(3, 10*time.Second, "test-internal-key")

	e.GET("/ws", func(c echo.Context) error {
		// Simulate auth middleware setting user_id
		c.Set("user_id", "test-user-123")
		return p.ProxyWebSocket(c, upstreamServer.URL, true)
	})

	gatewayServer := httptest.NewServer(e)
	defer gatewayServer.Close()

	// 3. Dial the gateway WebSocket
	wsURL := strings.Replace(gatewayServer.URL, "http", "ws", 1) + "/ws"
	dialer := websocket.Dialer{
		HandshakeTimeout: 5 * time.Second,
	}

	clientHeaders := make(http.Header)
	clientHeaders.Set("X-Custom-Header", "HelloGateway")

	conn, resp, err := dialer.Dial(wsURL, clientHeaders)
	require.NoError(t, err)
	defer conn.Close()
	defer resp.Body.Close()

	assert.Equal(t, http.StatusSwitchingProtocols, resp.StatusCode)

	// 4. Test communication
	testMsg := []byte("ping")
	err = conn.WriteMessage(websocket.TextMessage, testMsg)
	require.NoError(t, err)

	_, receivedMsg, err := conn.ReadMessage()
	require.NoError(t, err)
	assert.Equal(t, testMsg, receivedMsg)

	// 5. Verify headers forwarded to upstream
	require.NoError(t, upstreamErr)
	assert.Equal(t, "HelloGateway", upstreamReceivedHeaders.Get("X-Custom-Header"))
	assert.Equal(t, "test-user-123", upstreamReceivedHeaders.Get("X-User-ID"))
	assert.Equal(t, "test-internal-key", upstreamReceivedHeaders.Get("X-Internal-Key"))
}
