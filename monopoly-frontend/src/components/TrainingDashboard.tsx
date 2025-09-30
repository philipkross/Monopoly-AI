import { TrainingMetrics } from "@/types/monopoly";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, Target, Clock, Trophy } from "lucide-react";

interface TrainingDashboardProps {
  metrics: TrainingMetrics[];
  isTraining: boolean;
}

export function TrainingDashboard({ metrics, isTraining }: TrainingDashboardProps) {
  const latestMetrics = metrics[metrics.length - 1];
  const averageGameLength = metrics.length > 0 
    ? metrics.reduce((sum, m) => sum + m.gameLength, 0) / metrics.length 
    : 0;
  
  const winRates = metrics.length > 0 
    ? Array.from({ length: 4 }, (_, i) => 
        metrics.filter(m => m.winner === i).length / metrics.length * 100
      )
    : [0, 0, 0, 0];

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold">Training Dashboard</h3>
        <Badge variant={isTraining ? "default" : "secondary"}>
          {isTraining ? "Training" : "Idle"}
        </Badge>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 gap-3">
        <Card>
          <CardContent className="p-3">
            <div className="flex items-center gap-2">
              <Target className="w-4 h-4 text-primary" />
              <div>
                <p className="text-xs text-muted-foreground">Episodes</p>
                <p className="font-semibold">{metrics.length}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-3">
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4 text-primary" />
              <div>
                <p className="text-xs text-muted-foreground">Avg Length</p>
                <p className="font-semibold">{Math.round(averageGameLength)}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Latest Episode */}
      {latestMetrics && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <TrendingUp className="w-4 h-4" />
              Latest Episode #{latestMetrics.episode}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-muted-foreground">Winner:</span>
                <span className="font-semibold ml-1">Player {latestMetrics.winner + 1}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Length:</span>
                <span className="font-semibold ml-1">{latestMetrics.gameLength} turns</span>
              </div>
            </div>
            
            <div>
              <p className="text-xs text-muted-foreground mb-1">Player Rewards:</p>
              <div className="space-y-1">
                {latestMetrics.playerRewards.map((reward, i) => (
                  <div key={i} className="flex justify-between text-xs">
                    <span>Player {i + 1}:</span>
                    <span className={`font-semibold ${reward >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {reward > 0 ? '+' : ''}{reward.toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Win Rates */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <Trophy className="w-4 h-4" />
            Win Rates
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {winRates.map((rate, i) => (
              <div key={i} className="flex items-center gap-2">
                <div className={`w-3 h-3 rounded-full bg-piece-${['red', 'blue', 'green', 'yellow'][i]}`} />
                <span className="text-xs flex-1">Player {i + 1}</span>
                <span className="text-xs font-semibold">{rate.toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Recent Performance */}
      {metrics.length > 5 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Recent Performance</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-1">
              {metrics.slice(-5).reverse().map((metric, i) => (
                <div key={metric.episode} className="flex justify-between text-xs">
                  <span>Episode {metric.episode}:</span>
                  <span className="font-semibold">
                    Player {metric.winner + 1} wins
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}