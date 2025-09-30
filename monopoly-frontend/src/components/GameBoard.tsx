import { BoardSpace, Player, Property } from "@/types/monopoly";
import { PropertyCard } from "./PropertyCard";
import { PlayerPiece } from "./PlayerPiece";
import { cn } from "@/lib/utils";

interface GameBoardProps {
  players: Player[];
  properties: Property[];
  onPropertyClick?: (property: Property) => void;
}

const BOARD_SPACES: BoardSpace[] = [
  // Bottom row (right to left)
  { id: 0, name: "GO", type: "corner", position: { x: 90, y: 90, rotation: 0 } },
  { id: 1, name: "Mediterranean Ave", type: "property", position: { x: 82, y: 90, rotation: 0 }, group: "brown" },
  { id: 2, name: "Community Chest", type: "community", position: { x: 74, y: 90, rotation: 0 } },
  { id: 3, name: "Baltic Ave", type: "property", position: { x: 66, y: 90, rotation: 0 }, group: "brown" },
  { id: 4, name: "Income Tax", type: "tax", position: { x: 58, y: 90, rotation: 0 } },
  { id: 5, name: "Reading Railroad", type: "railroad", position: { x: 50, y: 90, rotation: 0 }, group: "railroad" },
  { id: 6, name: "Oriental Ave", type: "property", position: { x: 42, y: 90, rotation: 0 }, group: "light-blue" },
  { id: 7, name: "Chance", type: "chance", position: { x: 34, y: 90, rotation: 0 } },
  { id: 8, name: "Vermont Ave", type: "property", position: { x: 26, y: 90, rotation: 0 }, group: "light-blue" },
  { id: 9, name: "Connecticut Ave", type: "property", position: { x: 18, y: 90, rotation: 0 }, group: "light-blue" },
  
  // Left side (bottom to top)
  { id: 10, name: "Jail", type: "corner", position: { x: 10, y: 90, rotation: 0 } },
  { id: 11, name: "St. Charles Place", type: "property", position: { x: 10, y: 82, rotation: 90 }, group: "pink" },
  { id: 12, name: "Electric Company", type: "utility", position: { x: 10, y: 74, rotation: 90 }, group: "utility" },
  { id: 13, name: "States Ave", type: "property", position: { x: 10, y: 66, rotation: 90 }, group: "pink" },
  { id: 14, name: "Virginia Ave", type: "property", position: { x: 10, y: 58, rotation: 90 }, group: "pink" },
  { id: 15, name: "Pennsylvania Railroad", type: "railroad", position: { x: 10, y: 50, rotation: 90 }, group: "railroad" },
  { id: 16, name: "St. James Place", type: "property", position: { x: 10, y: 42, rotation: 90 }, group: "orange" },
  { id: 17, name: "Community Chest", type: "community", position: { x: 10, y: 34, rotation: 90 } },
  { id: 18, name: "Tennessee Ave", type: "property", position: { x: 10, y: 26, rotation: 90 }, group: "orange" },
  { id: 19, name: "New York Ave", type: "property", position: { x: 10, y: 18, rotation: 90 }, group: "orange" },
  
  // Top side (left to right)
  { id: 20, name: "Free Parking", type: "corner", position: { x: 10, y: 10, rotation: 0 } },
  { id: 21, name: "Kentucky Ave", type: "property", position: { x: 18, y: 10, rotation: 180 }, group: "red" },
  { id: 22, name: "Chance", type: "chance", position: { x: 26, y: 10, rotation: 180 } },
  { id: 23, name: "Indiana Ave", type: "property", position: { x: 34, y: 10, rotation: 180 }, group: "red" },
  { id: 24, name: "Illinois Ave", type: "property", position: { x: 42, y: 10, rotation: 180 }, group: "red" },
  { id: 25, name: "B&O Railroad", type: "railroad", position: { x: 50, y: 10, rotation: 180 }, group: "railroad" },
  { id: 26, name: "Atlantic Ave", type: "property", position: { x: 58, y: 10, rotation: 180 }, group: "yellow" },
  { id: 27, name: "Ventnor Ave", type: "property", position: { x: 66, y: 10, rotation: 180 }, group: "yellow" },
  { id: 28, name: "Water Works", type: "utility", position: { x: 74, y: 10, rotation: 180 }, group: "utility" },
  { id: 29, name: "Marvin Gardens", type: "property", position: { x: 82, y: 10, rotation: 180 }, group: "yellow" },
  
  // Right side (top to bottom)
  { id: 30, name: "Go To Jail", type: "corner", position: { x: 90, y: 10, rotation: 0 } },
  { id: 31, name: "Pacific Ave", type: "property", position: { x: 90, y: 18, rotation: 270 }, group: "green" },
  { id: 32, name: "North Carolina Ave", type: "property", position: { x: 90, y: 26, rotation: 270 }, group: "green" },
  { id: 33, name: "Community Chest", type: "community", position: { x: 90, y: 34, rotation: 270 } },
  { id: 34, name: "Pennsylvania Ave", type: "property", position: { x: 90, y: 42, rotation: 270 }, group: "green" },
  { id: 35, name: "Short Line Railroad", type: "railroad", position: { x: 90, y: 50, rotation: 270 }, group: "railroad" },
  { id: 36, name: "Chance", type: "chance", position: { x: 90, y: 58, rotation: 270 } },
  { id: 37, name: "Park Place", type: "property", position: { x: 90, y: 66, rotation: 270 }, group: "dark-blue" },
  { id: 38, name: "Luxury Tax", type: "tax", position: { x: 90, y: 74, rotation: 270 } },
  { id: 39, name: "Boardwalk", type: "property", position: { x: 90, y: 82, rotation: 270 }, group: "dark-blue" },
];

export function GameBoard({ players, properties, onPropertyClick }: GameBoardProps) {
  const getPropertyBySpaceId = (spaceId: number) => {
    return properties.find(p => p.id === spaceId);
  };

  return (
    <div className="relative w-full h-full bg-game-board rounded-lg shadow-[var(--shadow-game)] overflow-hidden">
      {/* Board background */}
      <div className="absolute inset-4 bg-primary/20 rounded-lg">
        <div className="absolute inset-2 bg-card rounded-lg flex items-center justify-center">
          <div className="text-center space-y-2">
            <h2 className="text-3xl font-bold text-primary">MONOPOLY</h2>
            <p className="text-muted-foreground">AI Training Environment</p>
          </div>
        </div>
      </div>

      {/* Board spaces */}
      {BOARD_SPACES.map((space) => {
        const property = space.type === 'property' || space.type === 'railroad' || space.type === 'utility' 
          ? getPropertyBySpaceId(space.id) 
          : null;
        
        return (
          <div
            key={space.id}
            className={cn(
              "absolute w-16 h-16",
              space.type === 'corner' && "w-20 h-20"
            )}
            style={{
              left: `${space.position.x}%`,
              top: `${space.position.y}%`,
              transform: `translate(-50%, -50%) rotate(${space.position.rotation}deg)`,
            }}
          >
            {property ? (
              <PropertyCard
                property={property}
                onClick={() => onPropertyClick?.(property)}
                className="w-full h-full"
              />
            ) : (
              <>
                {space.type === 'chance' && (
                  <div 
                    className="w-full h-full bg-gradient-to-br from-orange-400 to-red-500 flex flex-col items-center justify-center text-white text-xs font-bold rounded cursor-pointer hover:opacity-80"
                    onClick={() => console.log('Chance space clicked')}
                  >
                    <span>?</span>
                    <span className="text-[8px]">CHANCE</span>
                  </div>
                )}
                
                {space.type === 'community' && (
                  <div 
                    className="w-full h-full bg-gradient-to-br from-blue-400 to-purple-500 flex flex-col items-center justify-center text-white text-xs font-bold rounded cursor-pointer hover:opacity-80"
                    onClick={() => console.log('Community chest space clicked')}
                  >
                    <span>♣</span>
                    <span className="text-[8px]">CHEST</span>
                  </div>
                )}

                {space.type !== 'chance' && space.type !== 'community' && (
                  <div className={cn(
                    "w-full h-full bg-card border-2 border-border rounded-md flex items-center justify-center text-xs font-semibold text-center p-1",
                    space.type === 'corner' && "bg-accent text-accent-foreground",
                    space.type === 'tax' && "bg-red-100 text-red-800"
                  )}>
                    {space.name}
                  </div>
                )}
              </>
            )}
            
            {/* Player pieces on this space */}
            <div className="absolute -top-2 -right-2 flex flex-wrap gap-0.5">
              {players
                .filter(player => player.position === space.id)
                .map(player => (
                  <PlayerPiece
                    key={player.id}
                    player={player}
                    size="sm"
                  />
                ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}