import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Wifi, WifiOff, Play, Square } from "lucide-react";

interface ConnectionPanelProps {
  isConnected: boolean;
  onConnect: (config: ConnectionConfig) => void;
  onDisconnect: () => void;
  onStartTraining: () => void;
  onStopTraining: () => void;
  isTraining: boolean;
}

export interface ConnectionConfig {
  host: string;
  port: number;
  protocol: 'ws' | 'http';
}

export function ConnectionPanel({ 
  isConnected, 
  onConnect, 
  onDisconnect, 
  onStartTraining, 
  onStopTraining,
  isTraining 
}: ConnectionPanelProps) {
  const [config, setConfig] = useState<ConnectionConfig>({
    host: 'localhost',
    port: 8000,
    protocol: 'ws'
  });

  const handleConnect = () => {
    onConnect(config);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg flex items-center gap-2">
          {isConnected ? (
            <Wifi className="w-5 h-5 text-green-600" />
          ) : (
            <WifiOff className="w-5 h-5 text-red-600" />
          )}
          Python Backend Connection
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Connection Status */}
        <div className="flex items-center justify-between">
          <span className="text-sm">Status:</span>
          <Badge variant={isConnected ? "default" : "destructive"}>
            {isConnected ? "Connected" : "Disconnected"}
          </Badge>
        </div>

        {/* Connection Configuration */}
        {!isConnected && (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-2">
              <div>
                <Label htmlFor="host" className="text-xs">Host</Label>
                <Input
                  id="host"
                  value={config.host}
                  onChange={(e) => setConfig(prev => ({ ...prev, host: e.target.value }))}
                  placeholder="localhost"
                  className="h-8"
                />
              </div>
              <div>
                <Label htmlFor="port" className="text-xs">Port</Label>
                <Input
                  id="port"
                  type="number"
                  value={config.port}
                  onChange={(e) => setConfig(prev => ({ ...prev, port: parseInt(e.target.value) || 8000 }))}
                  placeholder="8000"
                  className="h-8"
                />
              </div>
            </div>
            
            <div>
              <Label htmlFor="protocol" className="text-xs">Protocol</Label>
              <select
                id="protocol"
                value={config.protocol}
                onChange={(e) => setConfig(prev => ({ ...prev, protocol: e.target.value as 'ws' | 'http' }))}
                className="w-full h-8 px-3 rounded-md border border-input bg-background text-sm"
              >
                <option value="ws">WebSocket</option>
                <option value="http">HTTP</option>
              </select>
            </div>

            <Button onClick={handleConnect} className="w-full" size="sm">
              Connect to Python Backend
            </Button>
          </div>
        )}

        {/* Connected Actions */}
        {isConnected && (
          <div className="space-y-3">
            <div className="text-sm text-muted-foreground">
              Connected to {config.protocol}://{config.host}:{config.port}
            </div>
            
            <div className="flex gap-2">
              {!isTraining ? (
                <Button onClick={onStartTraining} className="flex-1" size="sm">
                  <Play className="w-4 h-4 mr-1" />
                  Start Training
                </Button>
              ) : (
                <Button onClick={onStopTraining} variant="destructive" className="flex-1" size="sm">
                  <Square className="w-4 h-4 mr-1" />
                  Stop Training
                </Button>
              )}
              <Button onClick={onDisconnect} variant="outline" size="sm">
                Disconnect
              </Button>
            </div>
          </div>
        )}

        {/* Instructions */}
        <div className="text-xs text-muted-foreground space-y-1 border-t pt-3">
          <p><strong>Python Setup:</strong></p>
          <p>1. Install: pip install gymnasium numpy websockets</p>
          <p>2. Run your Monopoly Gymnasium environment</p>
          <p>3. Ensure WebSocket server is running on specified port</p>
        </div>
      </CardContent>
    </Card>
  );
}