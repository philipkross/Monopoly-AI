import { Card } from '@/types/monopoly';

export const CHANCE_CARDS: Card[] = [
  {
    id: 1,
    type: 'chance',
    title: 'Advance to GO',
    description: 'Advance to GO (Collect $200)',
    action: { type: 'move', position: 0, amount: 200 }
  },
  {
    id: 2,
    type: 'chance',
    title: 'Advance to Illinois Avenue',
    description: 'Advance to Illinois Avenue. If you pass GO, collect $200',
    action: { type: 'move', position: 24 }
  },
  {
    id: 3,
    type: 'chance',
    title: 'Advance to St. Charles Place',
    description: 'Advance to St. Charles Place. If you pass GO, collect $200',
    action: { type: 'move', position: 11 }
  },
  {
    id: 4,
    type: 'chance',
    title: 'Advance to Nearest Utility',
    description: 'Advance to the nearest utility. If unowned, you may buy it. If owned, pay 10x dice roll',
    action: { type: 'moveToNearest', spaceType: 'utility' }
  },
  {
    id: 5,
    type: 'chance',
    title: 'Advance to Nearest Railroad',
    description: 'Advance to the nearest railroad. If unowned, you may buy it. If owned, pay double rent',
    action: { type: 'moveToNearest', spaceType: 'railroad' }
  },
  {
    id: 6,
    type: 'chance',
    title: 'Bank Dividend',
    description: 'Bank pays you dividend of $50',
    action: { type: 'collect', amount: 50 }
  },
  {
    id: 7,
    type: 'chance',
    title: 'Get Out of Jail Free',
    description: 'Get Out of Jail Free card (Keep until used)',
    action: { type: 'getOutOfJail' }
  },
  {
    id: 8,
    type: 'chance',
    title: 'Go Back 3 Spaces',
    description: 'Go back 3 spaces',
    action: { type: 'move', position: -3 }
  },
  {
    id: 9,
    type: 'chance',
    title: 'Go to Jail',
    description: 'Go to Jail. Go directly to jail, do not pass GO, do not collect $200',
    action: { type: 'goToJail' }
  },
  {
    id: 10,
    type: 'chance',
    title: 'General Repairs',
    description: 'Make general repairs on all your property. For each house pay $25, for each hotel pay $100',
    action: { type: 'repairs', perHouse: 25, perHotel: 100 }
  },
  {
    id: 11,
    type: 'chance',
    title: 'Speeding Fine',
    description: 'Speeding fine $15',
    action: { type: 'pay', amount: 15 }
  },
  {
    id: 12,
    type: 'chance',
    title: 'Trip to Reading Railroad',
    description: 'Take a trip to Reading Railroad. If you pass GO, collect $200',
    action: { type: 'move', position: 5 }
  },
  {
    id: 13,
    type: 'chance',
    title: 'Trip to Boardwalk',
    description: 'Take a walk on the Boardwalk. Advance to Boardwalk',
    action: { type: 'move', position: 39 }
  },
  {
    id: 14,
    type: 'chance',
    title: 'Chairman of the Board',
    description: 'You have been elected Chairman of the Board. Pay each player $50',
    action: { type: 'payEach', amount: 50 }
  },
  {
    id: 15,
    type: 'chance',
    title: 'Building Loan Matures',
    description: 'Your building loan matures. Collect $150',
    action: { type: 'collect', amount: 150 }
  },
  {
    id: 16,
    type: 'chance',
    title: 'Crossword Competition',
    description: 'You have won a crossword competition. Collect $100',
    action: { type: 'collect', amount: 100 }
  }
];

export const COMMUNITY_CARDS: Card[] = [
  {
    id: 17,
    type: 'community',
    title: 'Advance to GO',
    description: 'Advance to GO (Collect $200)',
    action: { type: 'move', position: 0, amount: 200 }
  },
  {
    id: 18,
    type: 'community',
    title: 'Bank Error',
    description: 'Bank error in your favor. Collect $200',
    action: { type: 'collect', amount: 200 }
  },
  {
    id: 19,
    type: 'community',
    title: 'Doctor\'s Fee',
    description: 'Doctor\'s fee. Pay $50',
    action: { type: 'pay', amount: 50 }
  },
  {
    id: 20,
    type: 'community',
    title: 'Stock Sale',
    description: 'From sale of stock you get $50',
    action: { type: 'collect', amount: 50 }
  },
  {
    id: 21,
    type: 'community',
    title: 'Get Out of Jail Free',
    description: 'Get Out of Jail Free card (Keep until used)',
    action: { type: 'getOutOfJail' }
  },
  {
    id: 22,
    type: 'community',
    title: 'Go to Jail',
    description: 'Go to Jail. Go directly to jail, do not pass GO, do not collect $200',
    action: { type: 'goToJail' }
  },
  {
    id: 23,
    type: 'community',
    title: 'Holiday Fund',
    description: 'Holiday fund matures. Receive $100',
    action: { type: 'collect', amount: 100 }
  },
  {
    id: 24,
    type: 'community',
    title: 'Income Tax Refund',
    description: 'Income tax refund. Collect $20',
    action: { type: 'collect', amount: 20 }
  },
  {
    id: 25,
    type: 'community',
    title: 'Birthday',
    description: 'It is your birthday. Collect $10 from every player',
    action: { type: 'collectEach', amount: 10 }
  },
  {
    id: 26,
    type: 'community',
    title: 'Life Insurance',
    description: 'Life insurance matures. Collect $100',
    action: { type: 'collect', amount: 100 }
  },
  {
    id: 27,
    type: 'community',
    title: 'Hospital Fees',
    description: 'Pay hospital fees of $100',
    action: { type: 'pay', amount: 100 }
  },
  {
    id: 28,
    type: 'community',
    title: 'School Fees',
    description: 'Pay school fees of $50',
    action: { type: 'pay', amount: 50 }
  },
  {
    id: 29,
    type: 'community',
    title: 'Consultancy Fee',
    description: 'Receive $25 consultancy fee',
    action: { type: 'collect', amount: 25 }
  },
  {
    id: 30,
    type: 'community',
    title: 'Street Repairs',
    description: 'You are assessed for street repair. $40 per house, $115 per hotel',
    action: { type: 'repairs', perHouse: 40, perHotel: 115 }
  },
  {
    id: 31,
    type: 'community',
    title: 'Beauty Contest',
    description: 'You have won second prize in a beauty contest. Collect $10',
    action: { type: 'collect', amount: 10 }
  },
  {
    id: 32,
    type: 'community',
    title: 'Inheritance',
    description: 'You inherit $100',
    action: { type: 'collect', amount: 100 }
  }
];