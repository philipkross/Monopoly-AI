import { Property } from "@/types/monopoly";
import { cn } from "@/lib/utils";

interface PropertyCardProps {
  property: Property;
  onClick?: () => void;
  className?: string;
}

const getPropertyGroupColor = (group: string) => {
  const colorMap: Record<string, string> = {
    brown: "bg-property-brown",
    "light-blue": "bg-property-light-blue",
    pink: "bg-property-pink",
    orange: "bg-property-orange",
    red: "bg-property-red",
    yellow: "bg-property-yellow",
    green: "bg-property-green",
    "dark-blue": "bg-property-dark-blue",
    railroad: "bg-game-railroad text-white",
    utility: "bg-game-utility",
  };
  return colorMap[group] || "bg-muted";
};

export function PropertyCard({ property, onClick, className }: PropertyCardProps) {
  return (
    <div
      className={cn(
        "bg-card border border-border rounded-md overflow-hidden cursor-pointer transition-[var(--transition-smooth)] hover:shadow-[var(--shadow-property)]",
        property.owner !== undefined && "ring-2 ring-piece-" + ["red", "blue", "green", "yellow"][property.owner],
        className
      )}
      onClick={onClick}
    >
      {/* Property color band */}
      <div className={cn("h-3", getPropertyGroupColor(property.group))} />
      
      {/* Property content */}
      <div className="p-1 flex flex-col items-center justify-center h-[calc(100%-0.75rem)] text-xs">
        <div className="font-semibold text-center leading-tight">
          {property.name}
        </div>
        
        {/* Price */}
        <div className="text-game-money font-bold mt-0.5">
          ${property.price}
        </div>
        
        {/* Houses/Hotels */}
        {property.houses > 0 && (
          <div className="flex gap-0.5 mt-0.5">
            {Array.from({ length: property.houses }, (_, i) => (
              <div key={i} className="w-1.5 h-1.5 bg-green-600 rounded-sm" />
            ))}
          </div>
        )}
        
        {property.hotel && (
          <div className="w-2 h-2 bg-red-600 rounded-sm mt-0.5" />
        )}
        
        {/* Mortgaged indicator */}
        {property.mortgaged && (
          <div className="text-red-500 font-bold text-[8px] mt-0.5">MORT</div>
        )}
      </div>
    </div>
  );
}