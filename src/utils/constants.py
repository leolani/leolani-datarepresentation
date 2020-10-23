CAPITALIZED_TYPES = ['person']

NOT_TO_MENTION_TYPES = ['instance']

OBJECT_INFO = {u'home appliance': {'building': False, 'moving': False},
               u'swim cap': {'building': False, 'moving': True},
               u'office building': {'building': True, 'moving': False}, u'skis': {'building': False, 'moving': False},
               u'bell pepper': {'building': False, 'moving': True}, u'crocodile': {'building': False, 'moving': True},
               u'loveseat': {'building': False, 'moving': False},
               u'musical keyboard': {'building': False, 'moving': False}, u'chair': {'building': False, 'moving': True},
               u'dumbbell': {'building': False, 'moving': True}, u'milk': {'building': False, 'moving': False},
               u'ceiling fan': {'building': False, 'moving': True}, u'grape': {'building': False, 'moving': False},
               u'mixing bowl': {'building': False, 'moving': False}, u'tv': {'building': False, 'moving': False},
               u'swan': {'building': False, 'moving': True}, u'dairy': {'building': False, 'moving': False},
               u'artichoke': {'building': False, 'moving': False}, u'trousers': {'building': False, 'moving': True},
               u'jet ski': {'building': False, 'moving': False}, u'digital clock': {'building': False, 'moving': False},
               u'dragonfly': {'building': False, 'moving': True}, u'fax': {'building': False, 'moving': False},
               u'woman': {'building': False, 'moving': True}, u'ladle': {'building': False, 'moving': False},
               u'dessert': {'building': False, 'moving': False}, u'human hand': {'building': False, 'moving': True},
               u'screwdriver': {'building': False, 'moving': False}, u'vase': {'building': False, 'moving': False},
               u'spoon': {'building': False, 'moving': False}, u'tablet computer': {'building': False, 'moving': True},
               u'fireplace': {'building': False, 'moving': False}, u'segway': {'building': False, 'moving': False},
               u'stapler': {'building': False, 'moving': False}, u'whisk': {'building': False, 'moving': False},
               u'parrot': {'building': False, 'moving': True}, u'asparagus': {'building': False, 'moving': False},
               u'punching bag': {'building': False, 'moving': True}, u'dinosaur': {'building': False, 'moving': True},
               u'saxophone': {'building': False, 'moving': False}, u'mammal': {'building': False, 'moving': True},
               u'light bulb': {'building': False, 'moving': True}, u'snowmobile': {'building': False, 'moving': False},
               u'carnivore': {'building': False, 'moving': True}, u'bicycle': {'building': False, 'moving': False},
               u'tea': {'building': False, 'moving': False}, u'sports uniform': {'building': False, 'moving': True},
               u'lavender': {'building': False, 'moving': False}, u'horn': {'building': False, 'moving': False},
               u'nail': {'building': False, 'moving': False}, u'personal care': {'building': False, 'moving': False},
               u'fast food': {'building': False, 'moving': False}, u'panda': {'building': False, 'moving': True},
               u'power plugs and sockets': {'building': False, 'moving': True},
               u'clock': {'building': False, 'moving': False}, u'stool': {'building': False, 'moving': False},
               u'paddle': {'building': False, 'moving': False},
               u'medical equipment': {'building': False, 'moving': False},
               u'unicycle': {'building': False, 'moving': False}, u'bird': {'building': False, 'moving': True},
               u'facial tissue holder': {'building': False, 'moving': True},
               u'kangaroo': {'building': False, 'moving': True}, u'mule': {'building': False, 'moving': True},
               u'remote control': {'building': False, 'moving': False},
               u'indoor rower': {'building': False, 'moving': True}, u'sink': {'building': False, 'moving': False},
               u'pancake': {'building': False, 'moving': False}, u'goldfish': {'building': False, 'moving': True},
               u'dice': {'building': False, 'moving': False}, u'box': {'building': False, 'moving': False},
               u'boy': {'building': False, 'moving': True}, u'tortoise': {'building': False, 'moving': True},
               u'sports ball': {'building': False, 'moving': True}, u'hamster': {'building': False, 'moving': True},
               u'raccoon': {'building': False, 'moving': True}, u'pillow': {'building': False, 'moving': False},
               u'musical instrument': {'building': False, 'moving': True},
               u'roller skates': {'building': False, 'moving': True}, u'apple': {'building': False, 'moving': False},
               u'radish': {'building': False, 'moving': False}, u'traffic sign': {'building': False, 'moving': False},
               u'cricket ball': {'building': False, 'moving': True}, u'snowboard': {'building': False, 'moving': False},
               u'bicycle wheel': {'building': False, 'moving': False}, u'sun hat': {'building': False, 'moving': True},
               u'duck': {'building': False, 'moving': True}, u'hamburger': {'building': False, 'moving': False},
               u'kitchen appliance': {'building': False, 'moving': False},
               u'waffle iron': {'building': False, 'moving': False},
               u'bottle opener': {'building': False, 'moving': True},
               u'humidifier': {'building': False, 'moving': False}, u'frog': {'building': False, 'moving': True},
               u'camera': {'building': False, 'moving': False}, u'crab': {'building': False, 'moving': True},
               u'vehicle': {'building': False, 'moving': False}, u'light switch': {'building': False, 'moving': True},
               u'limousine': {'building': False, 'moving': False}, u'auto part': {'building': False, 'moving': False},
               u'human arm': {'building': False, 'moving': True}, u'ladybug': {'building': False, 'moving': True},
               u'sushi': {'building': False, 'moving': False}, u'door': {'building': False, 'moving': False},
               u'porch': {'building': False, 'moving': False}, u'envelope': {'building': False, 'moving': False},
               u'bronze sculpture': {'building': False, 'moving': False},
               u'missile': {'building': False, 'moving': False}, u'flag': {'building': False, 'moving': False},
               u'train': {'building': False, 'moving': False}, u'armadillo': {'building': False, 'moving': True},
               u'rabbit': {'building': False, 'moving': True}, u'salad': {'building': False, 'moving': False},
               u'car': {'building': False, 'moving': False}, u'chisel': {'building': False, 'moving': False},
               u'worm': {'building': False, 'moving': True}, u'cat': {'building': False, 'moving': True},
               u'donut': {'building': False, 'moving': False}, u'drill': {'building': False, 'moving': True},
               u'shark': {'building': False, 'moving': True}, u'headphones': {'building': False, 'moving': False},
               u'tap': {'building': False, 'moving': False}, u'dolphin': {'building': False, 'moving': True},
               u'parachute': {'building': False, 'moving': False}, u'drawer': {'building': False, 'moving': True},
               u'carrot': {'building': False, 'moving': False}, u'paper towel': {'building': False, 'moving': False},
               u'mixer': {'building': False, 'moving': False}, u'airplane': {'building': False, 'moving': False},
               u'woodpecker': {'building': False, 'moving': True}, u'clothing': {'building': False, 'moving': True},
               u'centipede': {'building': False, 'moving': True}, u'skirt': {'building': False, 'moving': True},
               u'magpie': {'building': False, 'moving': True}, u'rhinoceros': {'building': False, 'moving': True},
               u'lamp': {'building': False, 'moving': False}, u'sword': {'building': False, 'moving': False},
               u'elephant': {'building': False, 'moving': True}, u'frying pan': {'building': False, 'moving': False},
               u'cheetah': {'building': False, 'moving': True}, u'goat': {'building': False, 'moving': True},
               u'pizza': {'building': False, 'moving': False}, u'plant': {'building': False, 'moving': True},
               u'sandwich': {'building': False, 'moving': False}, u'cupboard': {'building': False, 'moving': False},
               u'briefcase': {'building': False, 'moving': False}, u'cocktail': {'building': False, 'moving': False},
               u'soap dispenser': {'building': False, 'moving': True},
               u'moths and butterflies': {'building': False, 'moving': True},
               u'toilet paper': {'building': False, 'moving': False}, u'taco': {'building': False, 'moving': True},
               u'cabbage': {'building': False, 'moving': False}, u'popcorn': {'building': False, 'moving': False},
               u'fedora': {'building': False, 'moving': True}, u'tomato': {'building': False, 'moving': False},
               u'oboe': {'building': False, 'moving': False},
               u'salt and pepper shakers': {'building': False, 'moving': True},
               u'croissant': {'building': False, 'moving': False}, u'wheelchair': {'building': False, 'moving': False},
               u'harbor seal': {'building': False, 'moving': True}, u'volleyball': {'building': False, 'moving': False},
               u'beehive': {'building': False, 'moving': False}, u'mango': {'building': False, 'moving': False},
               u'truck': {'building': False, 'moving': False}, u'wrench': {'building': False, 'moving': False},
               u'ambulance': {'building': False, 'moving': False}, u'necklace': {'building': False, 'moving': False},
               u'egg': {'building': False, 'moving': False}, u'wine': {'building': False, 'moving': False},
               u'antelope': {'building': False, 'moving': True}, u'owl': {'building': False, 'moving': True},
               u'muffin': {'building': False, 'moving': False}, u'canary': {'building': False, 'moving': True},
               u'spatula': {'building': False, 'moving': False}, u'jacuzzi': {'building': False, 'moving': False},
               u'motorcycle': {'building': False, 'moving': False}, u'oven': {'building': False, 'moving': False},
               u'keyboard': {'building': False, 'moving': False}, u'dagger': {'building': False, 'moving': False},
               u'pencil sharpener': {'building': False, 'moving': False},
               u'guacamole': {'building': False, 'moving': False}, u'accordion': {'building': False, 'moving': False},
               u'willow': {'building': False, 'moving': False}, u'houseplant': {'building': False, 'moving': False},
               u'backpack': {'building': False, 'moving': False},
               u'submarine sandwich': {'building': False, 'moving': False},
               u'window': {'building': False, 'moving': False},
               u'vehicle registration plate': {'building': False, 'moving': False},
               u'orange': {'building': False, 'moving': False}, u'tiara': {'building': False, 'moving': True},
               u'coffee': {'building': False, 'moving': False}, u'animal': {'building': False, 'moving': True},
               u'food processor': {'building': False, 'moving': True}, u'food': {'building': False, 'moving': False},
               u'tennis racket': {'building': False, 'moving': False},
               u'caterpillar': {'building': False, 'moving': True}, u'giraffe': {'building': False, 'moving': True},
               u'snake': {'building': False, 'moving': True}, u'horizontal bar': {'building': False, 'moving': False},
               u'bread': {'building': False, 'moving': False}, u'human beard': {'building': False, 'moving': True},
               u'convenience store': {'building': False, 'moving': False},
               u'filing cabinet': {'building': False, 'moving': False},
               u'jellyfish': {'building': False, 'moving': True}, u'lantern': {'building': False, 'moving': False},
               u'picture frame': {'building': False, 'moving': False},
               u'microwave oven': {'building': False, 'moving': False}, u'house': {'building': True, 'moving': False},
               u'fish': {'building': False, 'moving': True}, u'torch': {'building': False, 'moving': False},
               u'spider': {'building': False, 'moving': True}, u'cooking spray': {'building': False, 'moving': False},
               u'waffle': {'building': False, 'moving': False}, u'goose': {'building': False, 'moving': True},
               u'zebra': {'building': False, 'moving': True}, u'beetle': {'building': False, 'moving': True},
               u'girl': {'building': False, 'moving': True}, u'ice cream': {'building': True, 'moving': False},
               u'harp': {'building': False, 'moving': False},
               u'marine invertebrates': {'building': False, 'moving': True},
               u'flower': {'building': False, 'moving': False}, u'container': {'building': False, 'moving': False},
               u'lipstick': {'building': False, 'moving': False}, u'tennis ball': {'building': False, 'moving': True},
               u'alarm clock': {'building': False, 'moving': False}, u'fountain': {'building': False, 'moving': False},
               u'ladder': {'building': False, 'moving': False}, u'eagle': {'building': False, 'moving': True},
               u'umbrella': {'building': False, 'moving': False}, u'ipod': {'building': False, 'moving': False},
               u'cart': {'building': False, 'moving': False}, u'cookie': {'building': False, 'moving': True},
               u'van': {'building': False, 'moving': False}, u'hair dryer': {'building': False, 'moving': False},
               u'dishwasher': {'building': False, 'moving': True}, u'sea lion': {'building': False, 'moving': True},
               u'seahorse': {'building': False, 'moving': True}, u'snowplow': {'building': False, 'moving': False},
               u'skateboard': {'building': False, 'moving': False}, u'ostrich': {'building': False, 'moving': True},
               u'footwear': {'building': False, 'moving': True}, u'hedgehog': {'building': False, 'moving': True},
               u'lifejacket': {'building': False, 'moving': False}, u'scarf': {'building': False, 'moving': True},
               u'cheese': {'building': False, 'moving': False}, u'face powder': {'building': False, 'moving': True},
               u'coffee table': {'building': False, 'moving': False}, u'maracas': {'building': False, 'moving': False},
               u'sofa bed': {'building': False, 'moving': False}, u'scissors': {'building': False, 'moving': False},
               u'cell phone': {'building': False, 'moving': False}, u'maple': {'building': False, 'moving': False},
               u'sheep': {'building': False, 'moving': True}, u'computer monitor': {'building': False, 'moving': True},
               u'horse': {'building': False, 'moving': True}, u'toy': {'building': False, 'moving': True},
               u'monkey': {'building': False, 'moving': True}, u'man': {'building': False, 'moving': True},
               u'banana': {'building': False, 'moving': False}, u'bow and arrow': {'building': False, 'moving': False},
               u'starfish': {'building': False, 'moving': True}, u'common fig': {'building': False, 'moving': False},
               u'shelf': {'building': False, 'moving': False}, u'tool': {'building': False, 'moving': True},
               u'blue jay': {'building': False, 'moving': True}, u'rifle': {'building': False, 'moving': False},
               u'wardrobe': {'building': False, 'moving': False}, u'hand dryer': {'building': False, 'moving': True},
               u'stairs': {'building': False, 'moving': False}, u'television': {'building': False, 'moving': False},
               u'golf cart': {'building': False, 'moving': False}, u'tree': {'building': False, 'moving': True},
               u'french fries': {'building': False, 'moving': True}, u'bed': {'building': False, 'moving': False},
               u'bee': {'building': False, 'moving': True}, u'shower': {'building': False, 'moving': True},
               u'coat': {'building': False, 'moving': True}, u'human foot': {'building': False, 'moving': True},
               u'sculpture': {'building': False, 'moving': False}, u'coconut': {'building': False, 'moving': False},
               u'ant': {'building': False, 'moving': True}, u'juice': {'building': False, 'moving': False},
               u'pastry': {'building': False, 'moving': False}, u'microphone': {'building': False, 'moving': False},
               u'koala': {'building': False, 'moving': True}, u'shorts': {'building': False, 'moving': True},
               u'couch': {'building': False, 'moving': False}, u'bagel': {'building': False, 'moving': False},
               u'palm tree': {'building': False, 'moving': True}, u'human nose': {'building': False, 'moving': True},
               u'poster': {'building': False, 'moving': True}, u'banjo': {'building': False, 'moving': False},
               u'lily': {'building': False, 'moving': False}, u'honeycomb': {'building': False, 'moving': False},
               u'butterfly': {'building': False, 'moving': True}, u'printer': {'building': False, 'moving': True},
               u'kitchen & dining room table': {'building': False, 'moving': False},
               u'plumbing fixture': {'building': False, 'moving': True},
               u'invertebrate': {'building': False, 'moving': True}, u'polar bear': {'building': False, 'moving': True},
               u'handbag': {'building': False, 'moving': False},
               u'bowling equipment': {'building': False, 'moving': False},
               u'coin': {'building': False, 'moving': False}, u'refrigerator': {'building': False, 'moving': False},
               u'hot dog': {'building': False, 'moving': True}, u'seafood': {'building': False, 'moving': False},
               u'dog': {'building': False, 'moving': True}, u'measuring cup': {'building': False, 'moving': False},
               u'axe': {'building': False, 'moving': False}, u'mechanical fan': {'building': False, 'moving': True},
               u'barrel': {'building': False, 'moving': False}, u'skunk': {'building': False, 'moving': True},
               u'snowman': {'building': False, 'moving': False}, u'teddy bear': {'building': False, 'moving': True},
               u'infant bed': {'building': False, 'moving': True}, u'bomb': {'building': False, 'moving': False},
               u'kite': {'building': False, 'moving': True}, u'sparrow': {'building': False, 'moving': True},
               u'scoreboard': {'building': False, 'moving': False},
               u'training bench': {'building': False, 'moving': False},
               u'picnic basket': {'building': False, 'moving': False}, u'chicken': {'building': False, 'moving': True},
               u'blender': {'building': False, 'moving': False}, u'human head': {'building': False, 'moving': True},
               u'snack': {'building': False, 'moving': False}, u'shellfish': {'building': False, 'moving': True},
               u'tire': {'building': False, 'moving': False}, u'scorpion': {'building': False, 'moving': True},
               u'bust': {'building': False, 'moving': False}, u'balance beam': {'building': False, 'moving': True},
               u'piano': {'building': False, 'moving': False}, u'otter': {'building': False, 'moving': True},
               u'pumpkin': {'building': False, 'moving': False}, u'porcupine': {'building': False, 'moving': True},
               u'plate': {'building': False, 'moving': False}, u'sunglasses': {'building': False, 'moving': False},
               u'cannon': {'building': False, 'moving': False}, u'chopsticks': {'building': False, 'moving': False},
               u'watch': {'building': False, 'moving': True}, u'bear': {'building': False, 'moving': True},
               u'watermelon': {'building': False, 'moving': False},
               u'slow cooker': {'building': False, 'moving': False}, u'handgun': {'building': False, 'moving': False},
               u'bat': {'building': False, 'moving': True}, u'jaguar': {'building': False, 'moving': True},
               u'organ': {'building': False, 'moving': False}, u'sea turtle': {'building': False, 'moving': True},
               u'pretzel': {'building': False, 'moving': False}, u'perfume': {'building': False, 'moving': False},
               u'dog bed': {'building': False, 'moving': True}, u'pizza cutter': {'building': False, 'moving': True},
               u'computer keyboard': {'building': False, 'moving': True},
               u'human mouth': {'building': False, 'moving': True},
               u'billiard table': {'building': False, 'moving': False}, u'lion': {'building': False, 'moving': True},
               u'sewing machine': {'building': False, 'moving': True}, u'cattle': {'building': False, 'moving': True},
               u'racket': {'building': False, 'moving': False}, u'hammer': {'building': False, 'moving': False},
               u'cake stand': {'building': False, 'moving': False}, u'red panda': {'building': False, 'moving': True},
               u'closet': {'building': False, 'moving': False}, u'sombrero': {'building': False, 'moving': True},
               u'human hair': {'building': False, 'moving': True},
               u'computer mouse': {'building': False, 'moving': True}, u'swimwear': {'building': False, 'moving': True},
               u'mug': {'building': False, 'moving': True}, u'skyscraper': {'building': True, 'moving': False},
               u'wok': {'building': False, 'moving': False}, u'baked goods': {'building': False, 'moving': False},
               u'tent': {'building': False, 'moving': False}, u'cantaloupe': {'building': False, 'moving': False},
               u'bicycle helmet': {'building': False, 'moving': True},
               u'cocktail shaker': {'building': False, 'moving': True},
               u'hiking equipment': {'building': False, 'moving': False},
               u'cutting board': {'building': False, 'moving': False}, u'lynx': {'building': False, 'moving': True},
               u'toothbrush': {'building': False, 'moving': False}, u'aircraft': {'building': False, 'moving': False},
               u'wine glass': {'building': False, 'moving': False}, u'dress': {'building': False, 'moving': True},
               u'guitar': {'building': False, 'moving': False}, u'drum': {'building': False, 'moving': True},
               u'cow': {'building': False, 'moving': True}, u'baseball bat': {'building': False, 'moving': True},
               u'brassiere': {'building': False, 'moving': True}, u'bookcase': {'building': False, 'moving': False},
               u'wall clock': {'building': False, 'moving': False},
               u'door handle': {'building': False, 'moving': False},
               u'bathroom accessory': {'building': False, 'moving': True},
               u'swimming pool': {'building': False, 'moving': False},
               u'stretcher': {'building': False, 'moving': False}, u'pencil case': {'building': False, 'moving': True},
               u'jeans': {'building': False, 'moving': True}, u'rugby ball': {'building': False, 'moving': True},
               u'cabinetry': {'building': False, 'moving': False}, u'isopod': {'building': False, 'moving': True},
               u'table': {'building': False, 'moving': False}, u'trumpet': {'building': False, 'moving': False},
               u'boat': {'building': False, 'moving': False}, u'belt': {'building': False, 'moving': True},
               u'turkey': {'building': False, 'moving': True}, u'lighthouse': {'building': False, 'moving': False},
               u'watercraft': {'building': False, 'moving': False},
               u'rays and skates': {'building': False, 'moving': True},
               u'gondola': {'building': False, 'moving': False}, u'traffic light': {'building': False, 'moving': True},
               u'plastic bag': {'building': False, 'moving': True}, u'beer': {'building': False, 'moving': False},
               u'high heels': {'building': False, 'moving': True}, u'eraser': {'building': False, 'moving': False},
               u'curtain': {'building': False, 'moving': False},
               u'wood-burning stove': {'building': False, 'moving': False},
               u'squirrel': {'building': False, 'moving': True}, u'mushroom': {'building': False, 'moving': False},
               u'squash': {'building': False, 'moving': False}, u'marine mammal': {'building': False, 'moving': True},
               u'brown bear': {'building': False, 'moving': True}, u'bull': {'building': False, 'moving': True},
               u'corded phone': {'building': False, 'moving': False}, u'teapot': {'building': False, 'moving': False},
               u'drinking straw': {'building': False, 'moving': False}, u'tank': {'building': False, 'moving': False},
               u'hat': {'building': False, 'moving': True}, u'crutch': {'building': False, 'moving': False},
               u'lizard': {'building': False, 'moving': True}, u'ratchet': {'building': False, 'moving': False},
               u'castle': {'building': True, 'moving': False}, u'bathtub': {'building': False, 'moving': False},
               u'pressure cooker': {'building': False, 'moving': False},
               u'football helmet': {'building': False, 'moving': True}, u'canoe': {'building': False, 'moving': False},
               u'calculator': {'building': False, 'moving': True}, u'squid': {'building': False, 'moving': True},
               u'office supplies': {'building': False, 'moving': False},
               u'telephone': {'building': False, 'moving': False}, u'ring binder': {'building': False, 'moving': False},
               u'waste container': {'building': False, 'moving': False},
               u'nightstand': {'building': False, 'moving': False}, u'gas stove': {'building': False, 'moving': False},
               u'violin': {'building': False, 'moving': False}, u'mouse': {'building': False, 'moving': True},
               u'hair drier': {'building': False, 'moving': False}, u'helmet': {'building': False, 'moving': True},
               u'grinder': {'building': False, 'moving': False}, u'shirt': {'building': False, 'moving': True},
               u'balloon': {'building': False, 'moving': False}, u'toaster': {'building': False, 'moving': True},
               u'bowl': {'building': False, 'moving': False}, u'spice rack': {'building': False, 'moving': False},
               u'vegetable': {'building': False, 'moving': False}, u'wheel': {'building': False, 'moving': False},
               u'ball': {'building': False, 'moving': True}, u'snail': {'building': False, 'moving': True},
               u'pomegranate': {'building': False, 'moving': False}, u'drink': {'building': False, 'moving': False},
               u'leopard': {'building': False, 'moving': True}, u'binoculars': {'building': False, 'moving': False},
               u'cowboy hat': {'building': False, 'moving': True}, u'fruit': {'building': False, 'moving': False},
               u'cucumber': {'building': False, 'moving': False}, u'whale': {'building': False, 'moving': True},
               u'broccoli': {'building': False, 'moving': False}, u'street light': {'building': False, 'moving': True},
               u'burrito': {'building': False, 'moving': False}, u'surfboard': {'building': False, 'moving': False},
               u'shotgun': {'building': False, 'moving': False}, u'weapon': {'building': False, 'moving': False},
               u'person': {'building': False, 'moving': True}, u'bottle': {'building': False, 'moving': False},
               u'taxi': {'building': False, 'moving': False}, u'rocket': {'building': False, 'moving': False},
               u'camel': {'building': False, 'moving': True}, u'laptop': {'building': False, 'moving': False},
               u'desk': {'building': False, 'moving': False}, u'goggles': {'building': False, 'moving': False},
               u'fire hydrant': {'building': False, 'moving': False},
               u'cat furniture': {'building': False, 'moving': True},
               u'flashlight': {'building': False, 'moving': False}, u'sandal': {'building': False, 'moving': False},
               u'sunflower': {'building': False, 'moving': False}, u'cup': {'building': False, 'moving': False},
               u'luggage and bags': {'building': False, 'moving': True}, u'rose': {'building': False, 'moving': False},
               u'bench': {'building': False, 'moving': False}, u'grapefruit': {'building': False, 'moving': False},
               u'cosmetics': {'building': False, 'moving': False}, u'ski': {'building': False, 'moving': False},
               u'table tennis racket': {'building': False, 'moving': False},
               u'raven': {'building': False, 'moving': True},
               u'stationary bicycle': {'building': False, 'moving': False},
               u'ruler': {'building': False, 'moving': True}, u'miniskirt': {'building': False, 'moving': True},
               u'bathroom cabinet': {'building': False, 'moving': False},
               u'chainsaw': {'building': False, 'moving': False}, u'barge': {'building': False, 'moving': False},
               u'potted plant': {'building': False, 'moving': True}, u'falcon': {'building': False, 'moving': True},
               u'doughnut': {'building': False, 'moving': False}, u'tick': {'building': False, 'moving': True},
               u'furniture': {'building': False, 'moving': False}, u'towel': {'building': False, 'moving': False},
               u'glove': {'building': False, 'moving': True}, u'baseball glove': {'building': False, 'moving': True},
               u'tower': {'building': False, 'moving': False}, u'zucchini': {'building': False, 'moving': False},
               u'doll': {'building': False, 'moving': True}, u'crown': {'building': False, 'moving': True},
               u'lobster': {'building': False, 'moving': True}, u'candy': {'building': False, 'moving': False},
               u'mirror': {'building': False, 'moving': False}, u'billboard': {'building': False, 'moving': False},
               u'candle': {'building': False, 'moving': False}, u'tree house': {'building': True, 'moving': True},
               u'saucer': {'building': False, 'moving': False}, u'scale': {'building': False, 'moving': False},
               u'harmonica': {'building': False, 'moving': False}, u'kettle': {'building': False, 'moving': False},
               u'fox': {'building': False, 'moving': True}, u'microwave': {'building': False, 'moving': False},
               u'pen': {'building': False, 'moving': True}, u'treadmill': {'building': False, 'moving': False},
               u'earrings': {'building': False, 'moving': False}, u'human body': {'building': False, 'moving': True},
               u'knife': {'building': False, 'moving': False}, u'chest of drawers': {'building': False, 'moving': True},
               u'flute': {'building': False, 'moving': False}, u'pasta': {'building': False, 'moving': False},
               u'bidet': {'building': False, 'moving': False}, u'human ear': {'building': False, 'moving': True},
               u'beaker': {'building': False, 'moving': False}, u'whiteboard': {'building': False, 'moving': False},
               u'golf ball': {'building': False, 'moving': True}, u'shrimp': {'building': False, 'moving': True},
               u'hair spray': {'building': False, 'moving': False}, u'human eye': {'building': False, 'moving': True},
               u'glasses': {'building': False, 'moving': False}, u'heater': {'building': False, 'moving': False},
               u'reptile': {'building': False, 'moving': True}, u'tin can': {'building': False, 'moving': False},
               u'seat belt': {'building': False, 'moving': True}, u'stop sign': {'building': False, 'moving': False},
               u'coffeemaker': {'building': False, 'moving': False}, u'tableware': {'building': False, 'moving': False},
               u'mobile phone': {'building': False, 'moving': False}, u'syringe': {'building': False, 'moving': False},
               u'chime': {'building': False, 'moving': False},
               u'fashion accessory': {'building': False, 'moving': True},
               u'human leg': {'building': False, 'moving': True}, u'flying disc': {'building': False, 'moving': False},
               u'alpaca': {'building': False, 'moving': True}, u'submarine': {'building': False, 'moving': False},
               u'sock': {'building': False, 'moving': True}, u'wine rack': {'building': False, 'moving': False},
               u'winter melon': {'building': False, 'moving': False},
               u'can opener': {'building': False, 'moving': True}, u'harpsichord': {'building': False, 'moving': False},
               u'oyster': {'building': False, 'moving': True}, u'suit': {'building': False, 'moving': True},
               u'coffee cup': {'building': False, 'moving': False}, u'fork': {'building': False, 'moving': False},
               u'jug': {'building': False, 'moving': False}, u'bus': {'building': False, 'moving': False},
               u'cassette deck': {'building': False, 'moving': False}, u'pitcher': {'building': False, 'moving': True},
               u'tripod': {'building': False, 'moving': False}, u'diaper': {'building': False, 'moving': True},
               u'helicopter': {'building': False, 'moving': False},
               u'kitchen utensil': {'building': False, 'moving': False}, u'turtle': {'building': False, 'moving': True},
               u'kitchenware': {'building': False, 'moving': False}, u'penguin': {'building': False, 'moving': True},
               u'football': {'building': False, 'moving': False}, u'skull': {'building': False, 'moving': False},
               u'land vehicle': {'building': False, 'moving': True},
               u'strawberry': {'building': False, 'moving': False},
               u'christmas tree': {'building': False, 'moving': True}, u'cello': {'building': False, 'moving': False},
               u'suitcase': {'building': False, 'moving': False}, u'cake': {'building': False, 'moving': False},
               u'sports equipment': {'building': False, 'moving': True},
               u'toilet': {'building': False, 'moving': False}, u'deer': {'building': False, 'moving': True},
               u'kitchen knife': {'building': False, 'moving': False}, u'pig': {'building': False, 'moving': True},
               u'trombone': {'building': False, 'moving': False}, u'studio couch': {'building': False, 'moving': False},
               u'human face': {'building': False, 'moving': True}, u'cream': {'building': False, 'moving': False},
               u'serving tray': {'building': False, 'moving': False}, u'lemon': {'building': False, 'moving': False},
               u'peach': {'building': False, 'moving': True}, u'washing machine': {'building': False, 'moving': True},
               u'countertop': {'building': False, 'moving': False}, u'boot': {'building': False, 'moving': False},
               u'book': {'building': False, 'moving': False}, u'tie': {'building': False, 'moving': True},
               u'hippopotamus': {'building': False, 'moving': True}, u'tart': {'building': False, 'moving': True},
               u'frisbee': {'building': False, 'moving': False}, u'pineapple': {'building': False, 'moving': False},
               u'pear': {'building': False, 'moving': False}, u'insect': {'building': False, 'moving': True},
               u'paper cutter': {'building': False, 'moving': True}, u'building': {'building': True, 'moving': False},
               u'tiger': {'building': False, 'moving': True}, u'remote': {'building': False, 'moving': False},
               u'adhesive tape': {'building': False, 'moving': False}, u'potato': {'building': False, 'moving': False},
               u'band-aid': {'building': False, 'moving': False}, u'flowerpot': {'building': False, 'moving': False},
               u'jacket': {'building': False, 'moving': True}, u'parking meter': {'building': False, 'moving': False},
               u'stethoscope': {'building': False, 'moving': False}, u'platter': {'building': False, 'moving': False},
               u'window blind': {'building': False, 'moving': False}}
