import pandas as pd
import itertools
import random

# Define card ranks and suits
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['s', 'h', 'd', 'c']  # Spades, Hearts, Diamonds, Clubs

# Generate all unique two-card combinations
def generate_all_hands():
    all_hands = []
    for card1, card2 in itertools.combinations(itertools.product(RANKS, SUITS), 2):
        # Sort the hand to avoid duplicates like (A, K) and (K, A)
        if RANKS.index(card1[0]) > RANKS.index(card2[0]):
            hand = (card2, card1)
        else:
            hand = (card1, card2)
        all_hands.append(hand)
    return all_hands

# Define preflop ranges based on strategy
def get_preflop_ranges():
    preflop_ranges = {
        'Button': {
            'Loose': {
                'Pair': ['22', '33', '44', '55', '66', '77', '88', '99', 'TT', 'JJ', 'QQ', 'KK', 'AA'],
                'Broadway': ['AK', 'AQ', 'AJ', 'AT', 'KQ', 'KJ', 'KT', 'QJ', 'QT', 'JT'],
                'Suited Connector': ['65s', '54s'],
                'Offsuit Connector': ['T9o', 'JTo'],
                'Suited Ace': ['A2s', 'A3s', 'A4s', 'A5s', 'A6s', 'A7s', 'A8s', 'A9s', 'ATs', 'AJs', 'AQs', 'AKs'],
                'Offsuit Ace': ['A9o', 'ATo', 'AJo', 'AQo', 'AKo'],
                'Suited King': ['K8s', 'K9s', 'KTs', 'KJs', 'KQs'],
                'Offsuit King': ['KTo']
            },
            'Tight': {
                'Pair': ['55', '66', '77', '88', '99', 'TT', 'JJ', 'QQ', 'KK', 'AA'],
                'Broadway': ['AK', 'AQ', 'AJ', 'KQ', 'KJ'],
                'Suited Connector': ['98s', '87s'],
                'Offsuit Connector': [],
                'Suited Ace': ['A4s', 'A5s', 'A6s', 'A7s', 'A8s', 'A9s', 'ATs', 'AJs', 'AQs', 'AKs'],
                'Offsuit Ace': ['AJo', 'AQo', 'AKo'],
                'Suited King': ['K9s', 'KTs'],
                'Offsuit King': ['KTo']
            }
        },
        'Cut-Off': {
            'Loose': {
                'Pair': ['22', '33', '44', '55', '66', '77', '88', '99', 'TT', 'JJ', 'QQ', 'KK', 'AA'],
                'Broadway': ['AK', 'AQ', 'AJ', 'ATs', 'KQ', 'KJ', 'KT'],
                'Suited Connector': ['65s', '54s'],
                'Offsuit Connector': ['T9o', 'JTo'],
                'Suited Ace': ['A2s', 'A3s', 'A4s', 'A5s', 'A6s', 'A7s', 'A8s', 'A9s', 'ATs', 'AJs', 'AQs', 'AKs'],
                'Offsuit Ace': ['AJo', 'AQo', 'AKo'],
                'Suited King': ['K9s', 'KTs', 'KJs'],
                'Offsuit King': ['KTo']
            },
            'Tight': {
                'Pair': ['77', '88', '99', 'TT', 'JJ', 'QQ', 'KK', 'AA'],
                'Broadway': ['AK', 'AQ', 'AJ', 'KQ'],
                'Suited Connector': ['87s', '76s'],
                'Offsuit Connector': [],
                'Suited Ace': ['A5s', 'A6s', 'A7s', 'A8s', 'A9s', 'ATs', 'AJs', 'AQs', 'AKs'],
                'Offsuit Ace': ['AJo', 'AQo', 'AKo'],
                'Suited King': ['K9s', 'KTs', 'KJs'],
                'Offsuit King': ['KJo']
            }
        },
        'Early': {
            'Loose': {
                'Pair': ['66', '77', '88', '99', 'TT', 'JJ', 'QQ', 'KK', 'AA'],
                'Broadway': ['AK', 'AQ', 'AJ', 'KQ', 'KJ'],
                'Suited Connector': ['87s', '76s'],
                'Offsuit Connector': [],
                'Suited Ace': ['A3s', 'A4s', 'A5s', 'A6s', 'A7s', 'A8s', 'A9s', 'ATs', 'AJs', 'AQs', 'AKs'],
                'Offsuit Ace': ['AJo', 'AQo', 'AKo'],
                'Suited King': ['K9s', 'KTs', 'KJs'],
                'Offsuit King': ['KJo']
            },
            'Tight': {
                'Pair': ['88', '99', 'TT', 'JJ', 'QQ', 'KK', 'AA'],
                'Broadway': ['AK', 'AQ'],
                'Suited Connector': ['98s', '87s'],
                'Offsuit Connector': [],
                'Suited Ace': ['A5s', 'A6s', 'A7s', 'A8s', 'A9s', 'ATs', 'AJs', 'AQs', 'AKs'],
                'Offsuit Ace': ['AQo', 'AKo'],
                'Suited King': ['KQs'],
                'Offsuit King': []
            }
        },
        'Blinds': {
            'Facing Raises': {
                'Loose': {
                    'Pair': ['22', '33', '44', '55', '66', '77', '88', '99', 'TT', 'JJ', 'QQ', 'KK', 'AA'],
                    'Broadway': ['AK', 'AQ', 'AJ', 'AT', 'KQ', 'KJ', 'QT'],
                    'Suited Connector': ['76s'],
                    'Offsuit Connector': [],
                    'Suited Ace': ['A2s', 'A3s', 'A4s', 'A5s', 'A6s', 'A7s', 'A8s', 'A9s', 'ATs', 'AJs', 'AQs', 'AKs'],
                    'Offsuit Ace': ['AJo', 'AQo', 'AKo'],
                    'Suited King': ['K8s', 'KTs', 'KJs', 'KQs'],
                    'Offsuit King': ['KTo']
                },
                'Tight': {
                    'Pair': ['77', '88', '99', 'TT', 'JJ', 'QQ', 'KK', 'AA'],
                    'Broadway': ['AK', 'AQ', 'KQ'],
                    'Suited Connector': ['98s', '87s'],
                    'Offsuit Connector': [],
                    'Suited Ace': ['A5s', 'A6s', 'A7s', 'A8s', 'A9s', 'ATs', 'AJs', 'AQs', 'AKs'],
                    'Offsuit Ace': ['AQo', 'AKo'],
                    'Suited King': ['KTs'],
                    'Offsuit King': []
                }
            },
            'Limped Pots': {
                'Loose': {
                    'Pair': ['22', '33', '44', '55', '66', '77', '88', '99', 'TT', 'JJ', 'QQ', 'KK', 'AA'],
                    'Broadway': ['AK', 'AQ', 'AJ', 'AT', 'KQ', 'KJ', 'QT', 'JT'],
                    'Suited Connector': ['54s', '65s'],
                    'Offsuit Connector': ['T9o', 'JTo'],
                    'Suited Ace': ['A2s', 'A3s', 'A4s', 'A5s', 'A6s', 'A7s', 'A8s', 'A9s', 'ATs', 'AJs', 'AQs', 'AKs'],
                    'Offsuit Ace': [],
                    'Suited King': ['K8s', 'K9s', 'KTs', 'KJs', 'KQs'],
                    'Offsuit King': ['KTo']
                },
                'Tight': {
                    'Pair': ['77', '88', '99', 'TT', 'JJ', 'QQ', 'KK', 'AA'],
                    'Broadway': ['AK', 'AQ', 'AJ', 'KQ'],
                    'Suited Connector': ['98s', '87s'],
                    'Offsuit Connector': [],
                    'Suited Ace': ['A5s', 'A6s', 'A7s', 'A8s', 'A9s', 'ATs', 'AJs', 'AQs', 'AKs'],
                    'Offsuit Ace': [],
                    'Suited King': ['K9s', 'KTs', 'KJs', 'KQs'],
                    'Offsuit King': []
                }
            },
            'Steal Defense': {
                'Loose': {
                    'Pair': ['22', '33', '44', '55', '66', '77', '88', '99', 'TT', 'JJ', 'QQ', 'KK', 'AA'],
                    'Broadway': ['AK', 'AQ', 'AJ', 'AT', 'KQ', 'KJ'],
                    'Suited Connector': ['65s', '54s'],
                    'Offsuit Connector': [],
                    'Suited Ace': ['A2s', 'A3s', 'A4s', 'A5s', 'A6s', 'A7s', 'A8s', 'A9s', 'ATs', 'AJs', 'AQs', 'AKs'],
                    'Offsuit Ace': ['AJo', 'AQo', 'AKo'],
                    'Suited King': ['K8s', 'K9s', 'KTs', 'KJs', 'KQs'],
                    'Offsuit King': ['KTo']
                },
                'Tight': {
                    'Pair': ['22', '33', '44', '55', '66', '77', '88', '99', 'TT', 'JJ', 'QQ', 'KK', 'AA'],
                    'Broadway': ['AK', 'AQ', 'AJ', 'AT', 'KQ', 'KJ'],
                    'Suited Connector': ['65s', '54s'],
                    'Offsuit Connector': [],
                    'Suited Ace': ['A2s', 'A3s', 'A4s', 'A5s', 'A6s', 'A7s', 'A8s', 'A9s', 'ATs', 'AJs', 'AQs', 'AKs'],
                    'Offsuit Ace': ['AJo', 'AQo', 'AKo'],
                    'Suited King': ['K8s', 'K9s', 'KTs', 'KJs', 'KQs'],
                    'Offsuit King': ['KTo']
                }
            }
        }
    }
    return preflop_ranges

# Map hand to Hand Type and Hand Category
def categorize_hand(hand):
    rank1, suit1 = hand[0]
    rank2, suit2 = hand[1]

    # Determine if suited
    suited = 'Suited' if suit1 == suit2 else 'Offsuit'

    # Determine if pair
    if rank1 == rank2:
        hand_type = 'Pair'
        hand_category = rank1 + rank1
        return hand_type, hand_category

    # Determine broadway
    broadway_ranks = ['A', 'K', 'Q', 'J', 'T']
    if rank1 in broadway_ranks and rank2 in broadway_ranks:
        hand_type = 'Broadway'
        hand_category = rank1 + rank2 if RANKS.index(rank1) > RANKS.index(rank2) else rank2 + rank1
        return hand_type, hand_category

    # Determine suited connector
    rank_indices = sorted([RANKS.index(rank1), RANKS.index(rank2)])
    gap = rank_indices[1] - rank_indices[0] - 1
    if suited and gap == 0:
        hand_type = 'Suited Connector'
        hand_category = rank1 + rank2 + 's'
        return hand_type, hand_category
    elif suited and 1 <= gap <= 2:
        hand_type = 'Suited Connector'
        hand_category = rank1 + rank2 + 's'
        return hand_type, hand_category

    # Determine offsuit connector
    if not suited and gap == 0:
        hand_type = 'Offsuit Connector'
        hand_category = rank1 + rank2 + 'o'
        return hand_type, hand_category
    elif not suited and 1 <= gap <= 2:
        hand_type = 'Offsuit Connector'
        hand_category = rank1 + rank2 + 'o'
        return hand_type, hand_category

    # Determine suited ace
    if 'A' in [rank1, rank2]:
        other_rank = rank2 if rank1 == 'A' else rank1
        if suited:
            hand_type = 'Suited Ace'
            hand_category = 'A' + other_rank + 's'
            return hand_type, hand_category
        else:
            hand_type = 'Offsuit Ace'
            hand_category = 'A' + other_rank + 'o'
            return hand_type, hand_category

    # Determine suited king
    if 'K' in [rank1, rank2]:
        other_rank = rank2 if rank1 == 'K' else rank1
        if suited:
            hand_type = 'Suited King'
            hand_category = 'K' + other_rank + 's'
            return hand_type, hand_category
        else:
            hand_type = 'Offsuit King'
            hand_category = 'K' + other_rank + 'o'
            return hand_type, hand_category

    # If none of the above, classify as Other (optional)
    hand_type = 'Other'
    hand_category = rank1 + rank2 + ('s' if suited else 'o')
    return hand_type, hand_category

# Assign actions based on preflop ranges
def assign_action(position, sub_strategy, strategy, hand_category, preflop_ranges):
    if position != 'Blinds':
        # For non-Blinds positions, sub_strategy is None
        position_strategy_ranges = preflop_ranges[position][strategy]
        action = determine_action(position_strategy_ranges, hand_category)
    else:
        # For Blinds, sub_strategy is required
        position_strategy_ranges = preflop_ranges[position][sub_strategy][strategy]
        action = determine_action(position_strategy_ranges, hand_category)
    return action

def determine_action(position_strategy_ranges, hand_category):
    # Define action priorities
    # Higher priority: Raise
    # Next: Call
    # Else: Fold
    # Modify this function to include Call logic as needed
    # For now, assign Raise if in ranges, else Fold
    for hand_type, categories in position_strategy_ranges.items():
        if hand_type in ['Pair', 'Pairs']:
            if hand_category in categories:
                return 'Raise'
        elif hand_type == 'Broadway':
            if hand_category in categories:
                return 'Raise'
        elif hand_type == 'Suited Connector':
            if hand_category in categories:
                return 'Raise'
        elif hand_type == 'Offsuit Connector':
            if hand_category in categories:
                return 'Raise'
        elif hand_type == 'Suited Ace':
            if any(hand_category.startswith(a) for a in categories):
                return 'Raise'
        elif hand_type == 'Offsuit Ace':
            if any(hand_category.startswith(a) for a in categories):
                return 'Raise'
        elif hand_type == 'Suited King':
            if any(hand_category.startswith(a) for a in categories):
                return 'Raise'
        elif hand_type == 'Offsuit King':
            if any(hand_category.startswith(a) for a in categories):
                return 'Raise'
    # If not in any raise range, decide to Fold or Call
    # Here, we'll default to Fold
    return 'Fold'

# Main function to generate dataset
def generate_poker_dataset():
    all_hands = generate_all_hands()
    preflop_ranges = get_preflop_ranges()

    data = []

    # Define positions and strategies
    positions = ['Button', 'Cut-Off', 'Early', 'Blinds']
    strategies = ['Loose', 'Tight']

    for position in positions:
        if position != 'Blinds':
            for strategy in strategies:
                position_strategy_ranges = preflop_ranges[position][strategy]

                for hand in all_hands:
                    hand_type, hand_category = categorize_hand(hand)

                    # Skip hands not covered in the strategy
                    if hand_type not in position_strategy_ranges:
                        continue

                    # Assign action based on the strategy
                    action = assign_action(position, None, strategy, hand_category, preflop_ranges)

                    # Append to data
                    data.append({
                        'Position': position,
                        'Strategy': strategy,
                        'HandType': hand_type,
                        'HandCategory': hand_category,
                        'Action': action
                    })
        else:
            # For Blinds, iterate over each sub_strategy
            sub_strategies = ['Facing Raises', 'Limped Pots', 'Steal Defense']
            for sub_strategy in sub_strategies:
                for strategy in strategies:
                    position_strategy_ranges = preflop_ranges[position][sub_strategy][strategy]

                    for hand in all_hands:
                        hand_type, hand_category = categorize_hand(hand)

                        # Skip hands not covered in the strategy
                        if hand_type not in position_strategy_ranges:
                            continue

                        # Assign action based on the strategy and sub_strategy
                        action = assign_action(position, sub_strategy, strategy, hand_category, preflop_ranges)

                        # Append to data
                        data.append({
                            'Position': position,
                            'SubStrategy': sub_strategy,
                            'Strategy': strategy,
                            'HandType': hand_type,
                            'HandCategory': hand_category,
                            'Action': action
                        })

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Optional: Remove duplicates or irrelevant entries
    df.drop_duplicates(inplace=True)

    # Save to CSV
    df.to_csv('preflop_poker_dataset.csv', index=False)

    print("Dataset generated and saved to 'preflop_poker_dataset.csv'.")
