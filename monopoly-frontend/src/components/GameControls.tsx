import { GameState, GameAction, Player } from "@/types/monopoly";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PlayerPiece } from "./PlayerPiece";
import { Dice6 } from "lucide-react";

interface GameControlsProps {
  gameState: GameState;
  onAction: (action: GameAction) => void;
  isConnected: boolean;
}

export function GameControls({ gameState, onAction, isConnected }: GameControlsProps) {
  const currentPlayer = gameState.players[gameState.currentPlayer];
  
  const handleRoll = () => {
    onAction({
      type: 'roll',
      playerId: currentPlayer.id
    });
  };

  const handleBuy = () => {
    onAction({
      type: 'buy', 
      playerId: currentPlayer.id
    });
  };

  const handlePass = () => {
    onAction({
      type: 'pass',
      playerId: currentPlayer.id
    });
  };

  return (
    <div className="space-y-4">
      {/* Connection Status */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Python Backend</CardTitle>
        </CardHeader>
        <CardContent>
          <Badge variant={isConnected ? "default" : "destructive"}>
            {isConnected ? "Connected" : "Disconnected"}
          </Badge>
          {!isConnected && (
            <p className="text-xs text-muted-foreground mt-1">
              Connect your Python Gymnasium environment
            </p>
          )}
        </CardContent>
      </Card>

      {/* Current Player */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Current Turn</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center gap-2">
            <PlayerPiece player={currentPlayer} size="md" />
            <div>
              <p className="font-semibold">{currentPlayer.name}</p>
              <p className="text-xs text-muted-foreground">
                {currentPlayer.isAI ? "AI Agent" : "Human Player"}
              </p>
            </div>
          </div>
          
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <span className="text-muted-foreground">Money:</span>
              <span className="font-semibold text-game-money ml-1">
                ${currentPlayer.money}
              </span>
            </div>
            <div>
              <span className="text-muted-foreground">Properties:</span>
              <span className="font-semibold ml-1">
                {currentPlayer.properties.length}
              </span>
            </div>
          </div>

          {currentPlayer.inJail && (
            <Badge variant="destructive" className="text-xs">
              In Jail ({currentPlayer.jailTurns} turns)
            </Badge>
          )}
        </CardContent>
      </Card>

      {/* Dice */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <Dice6 className="w-4 h-4" />
            Dice Roll
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2 mb-3">
            <div className="w-8 h-8 bg-card border-2 border-border rounded flex items-center justify-center font-bold">
              {gameState.dice[0]}
            </div>
            <div className="w-8 h-8 bg-card border-2 border-border rounded flex items-center justify-center font-bold">
              {gameState.dice[1]}
            </div>
            <div className="text-sm text-muted-foreground">
              Total: {gameState.dice[0] + gameState.dice[1]}
            </div>
          </div>
          
          {gameState.phase === 'rolling' && !currentPlayer.isAI && (
            <Button onClick={handleRoll} className="w-full" size="sm">
              Roll Dice
            </Button>
          )}
        </CardContent>
      </Card>

      {/* Actions */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Actions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {gameState.phase === 'buying' && !currentPlayer.isAI && (
              <>
                <Button onClick={handleBuy} className="w-full" size="sm">
                  Buy Property
                </Button>
                <Button onClick={handlePass} variant="outline" className="w-full" size="sm">
                  Pass
                </Button>
              </>
            )}
            
            {gameState.phase === 'jail' && !currentPlayer.isAI && (
              <Button onClick={() => onAction({ type: 'pay_jail', playerId: currentPlayer.id })} 
                      className="w-full" size="sm">
                Pay $50 (Get out of jail)
              </Button>
            )}

            {currentPlayer.isAI && (
              <div className="text-center text-sm text-muted-foreground py-2">
                AI is thinking...
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Game Phase */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Game Status</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 text-xs">
            <div>
              <span className="text-muted-foreground">Phase:</span>
              <Badge variant="outline" className="ml-2 text-xs">
                {gameState.phase}
              </Badge>
            </div>
            <div>
              <span className="text-muted-foreground">Turn:</span>
              <span className="font-semibold ml-2">{gameState.turn}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Houses:</span>
              <span className="ml-2">{gameState.houses}/32</span>
            </div>
            <div>
              <span className="text-muted-foreground">Hotels:</span>
              <span className="ml-2">{gameState.hotels}/12</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}