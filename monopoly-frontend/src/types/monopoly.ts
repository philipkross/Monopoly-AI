export interface Property {
  id: number;
  name: string;
  group: PropertyGroup;
  price: number;
  rent: number[];
  houseCost?: number;
  owner?: number;
  houses: number;
  hotel: boolean;
  mortgaged: boolean;
}

export type PropertyGroup = 
  | 'brown' 
  | 'light-blue' 
  | 'pink' 
  | 'orange' 
  | 'red' 
  | 'yellow' 
  | 'green' 
  | 'dark-blue' 
  | 'railroad' 
  | 'utility' 
  | 'special';

export interface Player {
  id: number;
  name: string;
  position: number;
  money: number;
  properties: number[];
  inJail: boolean;
  jailTurns: number;
  piece: 'car' | 'hat' | 'dog' | 'shoe';
  color: 'red' | 'blue' | 'green' | 'yellow';
  isAI: boolean;
}

export interface GameState {
  players: Player[];
  currentPlayer: number;
  phase: 'waiting' | 'rolling' | 'moving' | 'buying' | 'trading' | 'jail' | 'finished' | 'card_drawn';
  dice: [number, number];
  properties: Property[];
  houses: number;
  hotels: number;
  turn: number;
  winner?: number;
  drawnCard?: Card;
  chanceCards: Card[];
  communityCards: Card[];
  chanceDiscard: number[];
  communityDiscard: number[];
}

export interface Card {
  id: number;
  type: 'chance' | 'community';
  title: string;
  description: string;
  action: CardAction;
}

export interface CardAction {
  type: 'move' | 'pay' | 'collect' | 'payEach' | 'collectEach' | 'moveToNearest' | 'goToJail' | 'getOutOfJail' | 'repairs';
  amount?: number;
  position?: number;
  spaceType?: 'railroad' | 'utility';
  perHouse?: number;
  perHotel?: number;
}

export interface GameAction {
  type: 'roll' | 'buy' | 'pass' | 'pay_jail' | 'use_card' | 'trade' | 'mortgage' | 'unmortgage' | 'build' | 'sell' | 'draw_card' | 'acknowledge_card';
  playerId: number;
  data?: any;
}

export interface TrainingMetrics {
  episode: number;
  totalReward: number;
  gameLength: number;
  winner: number;
  playerRewards: number[];
  actionsPerPlayer: number[];
}

export interface BoardSpace {
  id: number;
  name: string;
  type: 'property' | 'railroad' | 'utility' | 'corner' | 'tax' | 'chance' | 'community' | 'jail';
  position: { x: number; y: number; rotation: number };
  group?: PropertyGroup;
}