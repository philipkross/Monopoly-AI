import { Player } from "@/types/monopoly";
import { cn } from "@/lib/utils";

interface PlayerPieceProps {
  player: Player;
  size?: "sm" | "md" | "lg";
  className?: string;
}

const getPieceIcon = (piece: string) => {
  const icons: Record<string, string> = {
    car: "🚗",
    hat: "🎩", 
    dog: "🐕",
    shoe: "👟"
  };
  return icons[piece] || "🎮";
};

export function PlayerPiece({ player, size = "md", className }: PlayerPieceProps) {
  const sizeClasses = {
    sm: "w-4 h-4 text-xs",
    md: "w-6 h-6 text-sm", 
    lg: "w-8 h-8 text-base"
  };

  return (
    <div
      className={cn(
        "rounded-full flex items-center justify-center font-semibold shadow-sm border-2 border-white",
        sizeClasses[size],
        `bg-piece-${player.color}`,
        player.inJail && "opacity-60",
        className
      )}
      title={`${player.name} (${player.isAI ? 'AI' : 'Human'})`}
    >
      <span className="filter drop-shadow-sm">
        {getPieceIcon(player.piece)}
      </span>
    </div>
  );
}