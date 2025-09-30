import { Card } from '@/types/monopoly';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';

interface CardDisplayProps {
  card: Card | null;
  isOpen: boolean;
  onAcknowledge: () => void;
}

export const CardDisplay = ({ card, isOpen, onAcknowledge }: CardDisplayProps) => {
  if (!card) return null;

  const getCardStyle = () => {
    return card.type === 'chance' 
      ? 'bg-gradient-to-br from-orange-400 to-red-500 text-white'
      : 'bg-gradient-to-br from-blue-400 to-purple-500 text-white';
  };

  return (
    <Dialog open={isOpen} onOpenChange={() => {}}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle className={`text-center py-4 rounded-lg ${getCardStyle()}`}>
            {card.type === 'chance' ? 'CHANCE' : 'COMMUNITY CHEST'}
          </DialogTitle>
        </DialogHeader>
        
        <div className="space-y-4 p-4">
          <h3 className="text-lg font-bold text-center">{card.title}</h3>
          <p className="text-sm text-center text-muted-foreground">
            {card.description}
          </p>
        </div>

        <DialogFooter>
          <Button onClick={onAcknowledge} className="w-full">
            Continue
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};