import { useState, useEffect } from "react";
import { GameBoard } from "@/components/GameBoard";
import { GameControls } from "@/components/GameControls";
import { PlayerStatus } from "@/components/PlayerStatus";
import { TrainingDashboard } from "@/components/TrainingDashboard";
import { ConnectionPanel, ConnectionConfig } from "@/components/ConnectionPanel";
import { CardDisplay } from "@/components/CardDisplay";
import { GameState, GameAction, TrainingMetrics, Property, Player } from "@/types/monopoly";
import { CHANCE_CARDS, COMMUNITY_CARDS } from "@/data/cards";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

// Mock initial game state
const createInitialGameState = (): GameState => {
  const players: Player[] = [
    { id: 0, name: "AI Agent 1", position: 0, money: 1500, properties: [], inJail: false, jailTurns: 0, piece: "car", color: "red", isAI: true },
    { id: 1, name: "AI Agent 2", position: 0, money: 1500, properties: [], inJail: false, jailTurns: 0, piece: "hat", color: "blue", isAI: true },
  ];

  const properties: Property[] = Array.from({ length: 40 }, (_, i) => ({
    id: i,
    name: `Property ${i}`,
    group: 'brown',
    price: 100 + i * 20,
    rent: [10, 50, 150, 450, 625, 750],
    houseCost: 50,
    houses: 0,
    hotel: false,
    mortgaged: false
  }));

  return {
    players,
    currentPlayer: 0,
    phase: 'rolling',
    dice: [1, 1],
    properties,
    houses: 32,
    hotels: 12,
    turn: 1,
    chanceCards: [...CHANCE_CARDS],
    communityCards: [...COMMUNITY_CARDS],
    chanceDiscard: [],
    communityDiscard: []
  };
};

const Index = () => {
  const [gameState, setGameState] = useState<GameState>(createInitialGameState());
  const [isConnected, setIsConnected] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetrics[]>([]);
  const [websocket, setWebsocket] = useState<WebSocket | null>(null);

  // WebSocket connection management
  const handleConnect = (config: ConnectionConfig) => {
    try {
      const wsUrl = `ws://${config.host}:${config.port}/ws`;

      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        setIsConnected(true);
        setWebsocket(ws);
        console.log('Connected to Python backend');
      };
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'game_state') {
          setGameState(data.state);
        } else if (data.type === 'training_metrics') {
          setTrainingMetrics(prev => [...prev, data.metrics]);
        }
      };
      
      ws.onclose = () => {
        setIsConnected(false);
        setWebsocket(null);
        console.log('Disconnected from Python backend');
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
      };
      
    } catch (error) {
      console.error('Failed to connect:', error);
    }
  };

  const handleDisconnect = () => {
    if (websocket) {
      websocket.close();
    }
  };

  const handleStartTraining = () => {
    if (websocket && isConnected) {
      websocket.send(JSON.stringify({ type: 'start_training' }));
      setIsTraining(true);
    }
  };

  const handleStopTraining = () => {
    if (websocket && isConnected) {
      websocket.send(JSON.stringify({ type: 'stop_training' }));
      setIsTraining(false);
    }
  };

  const handleGameAction = (action: GameAction) => {
    if (websocket && isConnected) {
      websocket.send(JSON.stringify({ type: 'game_action', action }));
    } else {
      // Simulate local action for demo
      console.log('Game action:', action);
      
      // Simple dice roll simulation
      if (action.type === 'roll') {
        setGameState(prev => ({
          ...prev,
          dice: [Math.floor(Math.random() * 6) + 1, Math.floor(Math.random() * 6) + 1],
          phase: 'moving'
        }));
      }
    }
  };

  const handlePropertyClick = (property: Property) => {
    console.log('Property clicked:', property);
  };

  const handleCardAcknowledge = () => {
    handleGameAction({
      type: 'acknowledge_card',
      playerId: gameState.currentPlayer
    });
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (websocket) {
        websocket.close();
      }
    };
  }, [websocket]);

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto p-4">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-primary mb-2">Monopoly Plus</h1>
          <p className="text-muted-foreground">AI Training Environment for Reinforcement Learning</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Game Board - Takes up most space */}
          <div className="lg:col-span-2">
            <Card className="h-[600px]">
              <CardContent className="p-4 h-full">
                <GameBoard 
                  players={gameState.players}
                  properties={gameState.properties}
                  onPropertyClick={handlePropertyClick}
                />
              </CardContent>
            </Card>
          </div>

          {/* Controls and Status */}
          <div className="lg:col-span-2 space-y-4">
            <Tabs defaultValue="controls" className="w-full">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="controls">Game</TabsTrigger>
                <TabsTrigger value="training">Training</TabsTrigger>
                <TabsTrigger value="connection">Connection</TabsTrigger>
              </TabsList>
              
              <TabsContent value="controls" className="space-y-4">
                <GameControls 
                  gameState={gameState}
                  onAction={handleGameAction}
                  isConnected={isConnected}
                />
                <PlayerStatus 
                  players={gameState.players}
                  properties={gameState.properties}
                />
              </TabsContent>
              
              <TabsContent value="training">
                <TrainingDashboard 
                  metrics={trainingMetrics}
                  isTraining={isTraining}
                />
              </TabsContent>
              
              <TabsContent value="connection">
                <ConnectionPanel
                  isConnected={isConnected}
                  onConnect={handleConnect}
                  onDisconnect={handleDisconnect}
                  onStartTraining={handleStartTraining}
                  onStopTraining={handleStopTraining}
                  isTraining={isTraining}
                />
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </div>

      <CardDisplay 
        card={gameState.drawnCard || null}
        isOpen={gameState.phase === 'card_drawn'}
        onAcknowledge={handleCardAcknowledge}
      />
    </div>
  );
};

export default Index;
