# these are the directory names matched with the idx used for resnet classifications
subset_index_to_details = { 
    0: ('n01440764', 'tench'),
    1: ('n01443537', 'goldfish'),
    2: ('n01484850', 'great_white_shark'),
    3: ('n01491361', 'tiger_shark'),
    4: ('n01494475', 'hammerhead'),
    5: ('n01496331', 'electric_ray'),
    6: ('n01498041', 'stingray'),
    7: ('n01514668', 'cock'),
    8: ('n01514859', 'hen'),
    11: ('n01531178', 'goldfinch'),
    14: ('n01537544', 'indigo_bunting'),
    16: ('n01560419', 'bulbul'),
    18: ('n01582220', 'magpie'),
    19: ('n01592084', 'chickadee'),
    20: ('n01601694', 'water_ouzel'),
    21: ('n01608432', 'kite'),
    22: ('n01614925', 'bald_eagle'),
    24: ('n01622779', 'great_grey_owl'),
    26: ('n01630670', 'common_newt'),
    28: ('n01632458', 'spotted_salamander'),
    29: ('n01632777', 'axolotl'),
    32: ('n01644900', 'tailed_frog'),
    33: ('n01664065', 'loggerhead'),
    34: ('n01665541', 'leatherback_turtle'),
    35: ('n01667114', 'mud_turtle'),
    36: ('n01667778', 'terrapin'),
    38: ('n01675722', 'banded_gecko'),
    39: ('n01677366', 'common_iguana'),
    41: ('n01685808', 'whiptail'),
    42: ('n01687978', 'agama'),
    46: ('n01693334', 'green_lizard'),
    48: ('n01695060', 'Komodo_dragon'),
    50: ('n01698640', 'American_alligator'),
    52: ('n01728572', 'thunder_snake'),
    54: ('n01729322', 'hognose_snake'),
    55: ('n01729977', 'green_snake'),
    56: ('n01734418', 'king_snake'),
    57: ('n01735189', 'garter_snake'),
    59: ('n01739381', 'vine_snake'),
    60: ('n01740131', 'night_snake'),
    61: ('n01742172', 'boa_constrictor'),
    64: ('n01749939', 'green_mamba'),
    65: ('n01751748', 'sea_snake'),
    66: ('n01753488', 'horned_viper'),
    67: ('n01755581', 'diamondback'),
    68: ('n01756291', 'sidewinder'),
    70: ('n01770081', 'harvestman'),
    71: ('n01770393', 'scorpion'),
    72: ('n01773157', 'black_and_gold_garden_spider'),
    73: ('n01773549', 'barn_spider'),
    74: ('n01773797', 'garden_spider'),
    75: ('n01774384', 'black_widow'),
    76: ('n01774750', 'tarantula'),
    77: ('n01775062', 'wolf_spider'),
    78: ('n01776313', 'tick'),
    80: ('n01795545', 'black_grouse'),
    81: ('n01796340', 'ptarmigan'),
    83: ('n01798484', 'prairie_chicken'),
    84: ('n01806143', 'peacock'),
    88: ('n01818515', 'macaw'),
    89: ('n01819313', 'sulphur-crested_cockatoo'),
    90: ('n01820546', 'lorikeet'),
    91: ('n01824575', 'coucal'),
    92: ('n01828970', 'bee_eater'),
    93: ('n01829413', 'hornbill'),
    94: ('n01833805', 'hummingbird'),
    96: ('n01843383', 'toucan'),
    97: ('n01847000', 'drake'),
    99: ('n01855672', 'goose'),
    100: ('n01860187', 'black_swan'),
    104: ('n01877812', 'wallaby'),
    106: ('n01883070', 'wombat'),
    107: ('n01910747', 'jellyfish'),
    108: ('n01914609', 'sea_anemone'),
    110: ('n01924916', 'flatworm'),
    111: ('n01930112', 'nematode'),
    112: ('n01943899', 'conch'),
    113: ('n01944390', 'snail'),
    115: ('n01950731', 'sea_slug'),
    116: ('n01955084', 'chiton'),
    117: ('n01968897', 'chambered_nautilus'),
    118: ('n01978287', 'Dungeness_crab'),
    119: ('n01978455', 'rock_crab'),
    123: ('n01984695', 'spiny_lobster'),
    124: ('n01985128', 'crayfish'),
    125: ('n01986214', 'hermit_crab'),
    127: ('n02002556', 'white_stork'),
    129: ('n02006656', 'spoonbill'),
    130: ('n02007558', 'flamingo'),
    133: ('n02011460', 'bittern'),
    134: ('n02012849', 'crane'),
    135: ('n02013706', 'limpkin'),
    137: ('n02018207', 'American_coot'),
    138: ('n02018795', 'bustard'),
    140: ('n02027492', 'red-backed_sandpiper'),
    141: ('n02028035', 'redshank'),
    143: ('n02037110', 'oystercatcher'),
    144: ('n02051845', 'pelican'),
    146: ('n02058221', 'albatross'),
    150: ('n02077923', 'sea_lion')
}

def get_imagenet_index_and_label(wnid_to_find, full_index_map):
    for index_str, details in full_index_map.items():
        if isinstance(details, (list, tuple)) and len(details) >= 2:
            wnid = details[0]
            human_name = details[1]
            if wnid == wnid_to_find:
                index_int = int(index_str)
                return index_int, human_name
    return None, None




