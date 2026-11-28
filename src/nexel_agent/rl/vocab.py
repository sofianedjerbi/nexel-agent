"""The body's vocabulary: the tile characters, items, creature kinds and achievements its inputs have a place for,
and the actions its outputs are, in the order its embedding rows, vector slots and action rows were learned (the game
as of nexel-worlds 2fbb117, which r37-r43 trained on). Not a game rule: the policy's own reading of the game, frozen
so that a new creature or recipe in the game does not shift every id after it (r44: four new creatures and a hob
moved 30 tile ids and 66 actions by one, and the body played a scrambled world with crossed wires).

What the game has and this does not: a tile char reads as unknown (0), an item, kind or achievement has no slot, and
a new action goes after these (rl/actions.py), masked for a policy that never had it (rl/npolicy.py). A run learns
them when they are appended here, at a run boundary (rl/train.py --init grows the rows): never inserted."""

from nexel_worlds.ores import RENAMED       # (the game's own renames, nexel-worlds#403: the same thing, a new name)


def now(name):
    """A frozen name as the game calls the same thing now: "sword" is "iron_sword", "craft_ingot" is
    "craft_iron_ingot" (a learned row or slot keeps its place and its meaning, under the game's new name: #94)."""
    if not isinstance(name, str):
        return name
    kind, _, rest = name.partition("_")
    if kind in ("craft", "collect", "place") and rest in RENAMED:
        return f"{kind}_{RENAMED[rest]}"
    return RENAMED.get(name, name)


CHARS = '!"#$%&\'()*+,-.:;=?@BCDEFGHIJKLMOPQRSTUVWXYZ^_bcefghiklmnopqrstuvwxyz~'
ITEMS = (
    'axe', 'bandana', 'basket', 'bed', 'berries', 'blue_slippers', 'boat', 'bread', 'campfire', 'carrot',
    'cauliflower', 'chair', 'chest', 'clock', 'cooked_meat', 'coop', 'denim', 'dirt', 'door', 'egg', 'eggplant',
    'fence', 'fiber', 'firefly_charm', 'flag', 'flame_charm', 'flower', 'flower_charm', 'flower_crown',
    'frost_charm', 'fruit', 'gate', 'gold_boots', 'gold_chestplate', 'gold_crown', 'gold_greaves', 'halo_charm',
    'hay', 'heart_charm', 'hoe', 'ingot', 'iron_boots', 'iron_chestplate', 'iron_greaves', 'iron_helmet', 'ladder',
    'leaf_charm', 'leather_boots', 'leather_chaps', 'leather_vest', 'lettuce', 'linen_shirt', 'linen_shorts',
    'mailbox', 'meat', 'milk', 'moss_kilt', 'moss_slippers', 'mossback_helm', 'mossback_plate', 'mushroom',
    'name_tag', 'ore', 'overalls', 'path', 'pickaxe', 'picnic_blanket', 'plank', 'plant', 'pumpkin', 'rain_boots',
    'red_scarf', 'red_sneakers', 'red_trousers', 'robe_skirt', 'roof', 'rope', 'rug', 'seed', 'shell_plate', 'sign',
    'sparkle_charm', 'spear', 'spore_charm', 'sporecap_hat', 'sprout', 'stairs', 'stone', 'stone_block',
    'stone_wall', 'straw_hat', 'sword', 'table', 'torch', 'trough', 'watering_can', 'well', 'wheat', 'wizard_hat',
    'wizard_robe', 'wood', 'wood_wall', 'wool_beanie', 'wool_sweater', 'workbench',
)
KINDS = (
    'chicken', 'cow', 'chick', 'calf', 'tusker', 'pincer', 'strawman', 'wisp', 'wyrm', 'mossback', 'sporeling',
    'agent',
)
KNOWN = (
    'collect_berries', 'collect_carrot', 'collect_cauliflower', 'collect_dirt', 'collect_egg', 'collect_eggplant',
    'collect_fiber', 'collect_flower', 'collect_fruit', 'collect_hay', 'collect_lettuce', 'collect_meat',
    'collect_milk', 'collect_mushroom', 'collect_ore', 'collect_pumpkin', 'collect_seed', 'collect_shell_plate',
    'collect_stone', 'collect_stone_block', 'collect_wheat', 'collect_wood', 'craft_workbench', 'craft_plank',
    'craft_roof', 'craft_ladder', 'craft_stairs', 'craft_stone', 'craft_path', 'craft_fence', 'craft_door',
    'craft_sign', 'craft_flag', 'craft_campfire', 'craft_torch', 'craft_rope', 'craft_name_tag', 'craft_axe',
    'craft_pickaxe', 'craft_hoe', 'craft_spear', 'craft_stone_wall', 'craft_chest', 'craft_cooked_meat',
    'craft_bread', 'craft_ingot', 'craft_sword', 'craft_watering_can', 'craft_bed', 'craft_chair', 'craft_table',
    'craft_clock', 'craft_rug', 'craft_plant', 'craft_gate', 'craft_wood_wall', 'craft_boat', 'craft_mailbox',
    'craft_well', 'craft_coop', 'craft_picnic_blanket', 'craft_basket', 'craft_hay', 'craft_trough',
    'craft_straw_hat', 'craft_sporecap_hat', 'craft_wizard_hat', 'craft_gold_crown', 'craft_iron_helmet',
    'craft_mossback_helm', 'craft_wool_beanie', 'craft_flower_crown', 'craft_bandana', 'craft_sprout',
    'craft_linen_shirt', 'craft_red_scarf', 'craft_wool_sweater', 'craft_overalls', 'craft_leather_vest',
    'craft_wizard_robe', 'craft_iron_chestplate', 'craft_gold_chestplate', 'craft_mossback_plate',
    'craft_linen_shorts', 'craft_denim', 'craft_red_trousers', 'craft_leather_chaps', 'craft_robe_skirt',
    'craft_iron_greaves', 'craft_gold_greaves', 'craft_moss_kilt', 'craft_leather_boots', 'craft_red_sneakers',
    'craft_rain_boots', 'craft_blue_slippers', 'craft_iron_boots', 'craft_gold_boots', 'craft_moss_slippers',
    'craft_sparkle_charm', 'craft_firefly_charm', 'craft_leaf_charm', 'craft_heart_charm', 'craft_flame_charm',
    'craft_frost_charm', 'craft_halo_charm', 'craft_spore_charm', 'craft_flower_charm', 'place_basket', 'place_bed',
    'place_boat', 'place_campfire', 'place_chair', 'place_chest', 'place_clock', 'place_coop', 'place_dirt',
    'place_door', 'place_fence', 'place_flag', 'place_flower', 'place_gate', 'place_hay', 'place_ladder',
    'place_mailbox', 'place_path', 'place_picnic_blanket', 'place_plank', 'place_plant', 'place_roof', 'place_rug',
    'place_seed', 'place_sign', 'place_stairs', 'place_stone_block', 'place_stone_wall', 'place_table',
    'place_trough', 'place_well', 'place_wood_wall', 'place_workbench', 'eat_wheat', 'eat_berries', 'eat_mushroom',
    'eat_lettuce', 'eat_meat', 'eat_egg', 'eat_fruit', 'eat_carrot', 'eat_milk', 'eat_cauliflower', 'eat_eggplant',
    'eat_pumpkin', 'eat_cooked_meat', 'eat_bread', 'defeat_chicken', 'defeat_cow', 'defeat_chick', 'defeat_calf',
    'defeat_tusker', 'defeat_pincer', 'defeat_strawman', 'defeat_wyrm', 'defeat_mossback', 'defeat_sporeling',
    'till', 'plant', 'water_crop', 'feed_animal', 'milk_cow', 'sleep_bed', 'wear_any', 'wear_armor',
    'place_5_kinds', 'place_10_kinds', 'win_hurt', 'defeat_player', 'fought_armed',
)
ACTIONS = (
    ('noop', None, None), ('step', 'N', None), ('step', 'E', None), ('step', 'S', None), ('step', 'W', None),
    ('jump', 'N', None), ('jump', 'E', None), ('jump', 'S', None), ('jump', 'W', None), ('dig', 'N', None), ('dig',
    'E', None), ('dig', 'S', None), ('dig', 'W', None), ('dig', 'N', 'axe'), ('dig', 'E', 'axe'), ('dig', 'S',
    'axe'), ('dig', 'W', 'axe'), ('dig', 'N', 'pickaxe'), ('dig', 'E', 'pickaxe'), ('dig', 'S', 'pickaxe'), ('dig',
    'W', 'pickaxe'), ('dig', 'N', 'hoe'), ('dig', 'E', 'hoe'), ('dig', 'S', 'hoe'), ('dig', 'W', 'hoe'), ('dig_at',
    'N', 1), ('dig_at', 'E', 1), ('dig_at', 'S', 1), ('dig_at', 'W', 1), ('dig_at', 'N', 2), ('dig_at', 'E', 2),
    ('dig_at', 'S', 2), ('dig_at', 'W', 2), ('climb', 'N', None), ('climb', 'E', None), ('climb', 'S', None),
    ('climb', 'W', None), ('dig_down', None, None), ('pick_up', None, None), ('attack', None, None), ('water', None,
    None), ('sleep', None, None), ('feed', None, None), ('milk', None, None), ('wear', None, None), ('place', None,
    'basket'), ('place', None, 'bed'), ('place', None, 'boat'), ('place', None, 'campfire'), ('place', None,
    'chair'), ('place', None, 'chest'), ('place', None, 'clock'), ('place', None, 'coop'), ('place', None, 'dirt'),
    ('place', None, 'door'), ('place', None, 'fence'), ('place', None, 'flag'), ('place', None, 'flower'), ('place',
    None, 'gate'), ('place', None, 'hay'), ('place', None, 'ladder'), ('place', None, 'mailbox'), ('place', None,
    'path'), ('place', None, 'picnic_blanket'), ('place', None, 'plank'), ('place', None, 'plant'), ('place', None,
    'roof'), ('place', None, 'rug'), ('place', None, 'seed'), ('place', None, 'sign'), ('place', None, 'stairs'),
    ('place', None, 'stone_block'), ('place', None, 'stone_wall'), ('place', None, 'table'), ('place', None,
    'trough'), ('place', None, 'well'), ('place', None, 'wood_wall'), ('place', None, 'workbench'), ('craft', None,
    'workbench'), ('craft', None, 'plank'), ('craft', None, 'roof'), ('craft', None, 'ladder'), ('craft', None,
    'stairs'), ('craft', None, 'stone'), ('craft', None, 'path'), ('craft', None, 'fence'), ('craft', None, 'door'),
    ('craft', None, 'sign'), ('craft', None, 'flag'), ('craft', None, 'campfire'), ('craft', None, 'torch'),
    ('craft', None, 'rope'), ('craft', None, 'name_tag'), ('craft', None, 'axe'), ('craft', None, 'pickaxe'),
    ('craft', None, 'hoe'), ('craft', None, 'spear'), ('craft', None, 'stone_wall'), ('craft', None, 'chest'),
    ('craft', None, 'cooked_meat'), ('craft', None, 'bread'), ('craft', None, 'ingot'), ('craft', None, 'sword'),
    ('craft', None, 'watering_can'), ('craft', None, 'bed'), ('craft', None, 'chair'), ('craft', None, 'table'),
    ('craft', None, 'clock'), ('craft', None, 'rug'), ('craft', None, 'plant'), ('craft', None, 'gate'), ('craft',
    None, 'wood_wall'), ('craft', None, 'boat'), ('craft', None, 'mailbox'), ('craft', None, 'well'), ('craft',
    None, 'coop'), ('craft', None, 'picnic_blanket'), ('craft', None, 'basket'), ('craft', None, 'hay'), ('craft',
    None, 'trough'), ('craft', None, 'straw_hat'), ('craft', None, 'sporecap_hat'), ('craft', None, 'wizard_hat'),
    ('craft', None, 'gold_crown'), ('craft', None, 'iron_helmet'), ('craft', None, 'mossback_helm'), ('craft', None,
    'wool_beanie'), ('craft', None, 'flower_crown'), ('craft', None, 'bandana'), ('craft', None, 'sprout'),
    ('craft', None, 'linen_shirt'), ('craft', None, 'red_scarf'), ('craft', None, 'wool_sweater'), ('craft', None,
    'overalls'), ('craft', None, 'leather_vest'), ('craft', None, 'wizard_robe'), ('craft', None,
    'iron_chestplate'), ('craft', None, 'gold_chestplate'), ('craft', None, 'mossback_plate'), ('craft', None,
    'linen_shorts'), ('craft', None, 'denim'), ('craft', None, 'red_trousers'), ('craft', None, 'leather_chaps'),
    ('craft', None, 'robe_skirt'), ('craft', None, 'iron_greaves'), ('craft', None, 'gold_greaves'), ('craft', None,
    'moss_kilt'), ('craft', None, 'leather_boots'), ('craft', None, 'red_sneakers'), ('craft', None, 'rain_boots'),
    ('craft', None, 'blue_slippers'), ('craft', None, 'iron_boots'), ('craft', None, 'gold_boots'), ('craft', None,
    'moss_slippers'), ('craft', None, 'sparkle_charm'), ('craft', None, 'firefly_charm'), ('craft', None,
    'leaf_charm'), ('craft', None, 'heart_charm'), ('craft', None, 'flame_charm'), ('craft', None, 'frost_charm'),
    ('craft', None, 'halo_charm'), ('craft', None, 'spore_charm'), ('craft', None, 'flower_charm'), ('eat', None,
    'wheat'), ('eat', None, 'berries'), ('eat', None, 'mushroom'), ('eat', None, 'lettuce'), ('eat', None, 'meat'),
    ('eat', None, 'egg'), ('eat', None, 'fruit'), ('eat', None, 'carrot'), ('eat', None, 'milk'), ('eat', None,
    'cauliflower'), ('eat', None, 'eggplant'), ('eat', None, 'pumpkin'), ('eat', None, 'cooked_meat'), ('eat', None,
    'bread'), ('say', None, None), ('give', None, None), ('ask', None, None), ('approach', None, None), ('attack',
    None, 1), ('attack', None, 2), ('attack', None, 'back'), ('ring', None, None), ('pursue', None, None),
    # (ACTIONS_V4, encode.py: what r57 trained with, the game at 31963d92: its tool tiers and ores, the hob, build)
    ('craft', None, 'wood_axe'), ('craft', None, 'wood_pickaxe'), ('craft', None, 'iron_axe'), ('craft', None,
    'iron_pickaxe'), ('craft', None, 'titanium_axe'), ('craft', None, 'titanium_pickaxe'), ('craft', None,
    'titanium_sword'), ('craft', None, 'iron_ingot'), ('craft', None, 'copper_ingot'), ('craft', None,
    'titanium_ingot'), ('craft', None, 'gold_coin'), ('craft', None, 'wire'), ('craft', None, 'hob'), ('build', None,
    None),
)
# Pack items that came after ITEMS was learned (nexel-worlds#41's ores, ingots and tool tiers; the hob, the Eclipse
# Core): a block of their own at the vec's end (encode.py ITEMS_V2), not appended to ITEMS, whose block is in its middle.
# ("ore" and "ingot" in ITEMS are the game's iron_ore and iron_ingot now: theirs are never set)
ITEMS_V2 = (
    'coal', 'copper_ore', 'iron_ore', 'gold_ore', 'titanium_ore', 'copper_ingot', 'iron_ingot', 'titanium_ingot',
    'gold_coin', 'wire', 'wood_axe', 'wood_pickaxe', 'iron_axe', 'iron_pickaxe', 'titanium_axe', 'titanium_pickaxe',
    'titanium_sword', 'hob', 'eclipse_core',
)
