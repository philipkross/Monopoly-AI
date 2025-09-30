import { Player, Property } from "@/types/monopoly";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PlayerPiece } from "./PlayerPiece";
import { Progress } from "@/components/ui/progress";

interface PlayerStatusProps {
  players: Player[];
  properties: Property[];
}

export function PlayerStatus({ players, properties }: PlayerStatusProps) {
  const getPlayerNetWorth = (player: Player) => {
    const propertyValue = player.properties.reduce((total, propId) => {
      const property = properties.find(p => p.id === propId);
      return total + (property ? property.price : 0);
    }, 0);
    return player.money + propertyValue;
  };

  const totalWealth = players.reduce((sum, player) => sum + getPlayerNetWorth(player), 0);

  return (
    <div className="space-y-3">
      <h3 className="font-semibold text-sm">Players</h3>
      
      {players.map((player) => {
        const netWorth = getPlayerNetWorth(player);
        const wealthPercentage = totalWealth > 0 ? (netWorth / totalWealth) * 100 : 0;
        
        return (
          <Card key={player.id} className="p-3">
            <div className="flex items-center gap-3 mb-2">
              <PlayerPiece player={player} size="md" />
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-sm">{player.name}</span>
                  {player.isAI && (
                    <Badge variant="secondary" className="text-xs">AI</Badge>
                  )}
                </div>
                <div className="text-xs text-muted-foreground">
                  Net Worth: <span className="font-semibold text-game-money">${netWorth}</span>
                </div>
              </div>
            </div>

            <Progress value={wealthPercentage} className="h-2 mb-2" />

            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-muted-foreground">Cash:</span>
                <span className="font-semibold text-game-money ml-1">
                  ${player.money}
                </span>
              </div>
              <div>
                <span className="text-muted-foreground">Props:</span>
                <span className="font-semibold ml-1">
                  {player.properties.length}
                </span>
              </div>
            </div>

            {player.inJail && (
              <Badge variant="destructive" className="text-xs mt-2">
                In Jail
              </Badge>
            )}
          </Card>
        );
      })}
    </div>
  );
}