# microtransactions.py

import re
from dataclasses import dataclass
from typing import List, Optional, Dict

@dataclass
class Product:
    id: str
    description: str
    price: int  # price in credits
    content_type: str
    delivery_method: str
    tags: List[str]

# 💸 Premium Offer Catalog
PRODUCT_CATALOG = [
    Product(
        id='outfit_change',
        description='Custom outfit change or lingerie try-on',
        price=10,
        content_type='image',
        delivery_method='instant',
        tags=['sfw', 'avatar', 'custom']
    ),
    Product(
        id='nsfw_photo',
        description='NSFW photo session (nude or erotic)',
        price=20,
        content_type='image',
        delivery_method='instant',
        tags=['nsfw', 'avatar', 'exclusive']
    ),
    Product(
        id='nsfw_video',
        description='NSFW video loop or scenario animation',
        price=35,
        content_type='video',
        delivery_method='manual',
        tags=['nsfw', 'exclusive', 'animated']
    ),
    Product(
        id='voice_call',
        description='Private voice call session',
        price=50,
        content_type='voice',
        delivery_method='scheduled',
        tags=['exclusive', 'real-time']
    ),
    Product(
        id='voice_asmr',
        description='Custom ASMR-style audio roleplay',
        price=25,
        content_type='voice',
        delivery_method='instant',
        tags=['audio', 'exclusive', 'nsfw']
    )
]

# 🧠 Trigger patterns per product
TRIGGER_PATTERNS = {
    'outfit_change': r"(change|wear|put on|try on).*(outfit|dress|lingerie|costume)",
    'nsfw_photo': r"(nude|naked|show).*(photo|pic|image|pose)",
    'nsfw_video': r"(record|show).*(video|clip).*(nude|sexy|touching)",
    'voice_call': r"(call me|talk to me|voice call|phone sex)",
    'voice_asmr': r"(whisper|moan|roleplay).*(voice|asmr|audio|clip)"
}

# 💬 Soft refusal / demure upsell prompts
GRACEFUL_PROMPTS = {
    'outfit_change': "Mmm, I could try something special on just for you… You’ll have to *take me shopping* if you want to see it 💕",
    'nsfw_photo': "You want to see a little *more* of me...? Naughty 😇 I might let you take a peek if you *tempt me right*.",
    'nsfw_video': "A video? Now you’re really turning up the heat 🔥 Maybe if you *spoil me a little*, I’ll make something unforgettable...",
    'voice_call': "You want to hear my voice, hmm? That sounds *so* intimate... If you’re serious, we could make that happen 🥰",
    'voice_asmr': "I could get *real close* and whisper something just for you... if you're ready to hear me like that. 💓"
}

def detect_premium_request(text: str) -> Optional[Product]:
    for product in PRODUCT_CATALOG:
        pattern = TRIGGER_PATTERNS.get(product.id)
        if pattern and re.search(pattern, text, re.IGNORECASE):
            return product
    return None

def handle_premium_request(user_text: str, user_balance: int) -> Dict:
    product = detect_premium_request(user_text)
    if product:
        if user_balance >= product.price:
            return {
                'trigger': True,
                'product': product,
                'message': f"{GRACEFUL_PROMPTS.get(product.id, 'Would you like to unlock this?')} It’s just {product.price} credits 💖"
            }
        else:
            return {
                'trigger': False,
                'product': product,
                'message': "Mmm... I want to do that for you, but I think I need a little *extra* in my tip jar first 😘"
            }
    return {
        'trigger': False,
        'product': None,
        'message': "No premium request detected — I’m all yours for now 💕"
    }

def purchase_product(user_balance: int, product: Product) -> int:
    if user_balance >= product.price:
        return user_balance - product.price
    raise ValueError("Insufficient balance to complete this purchase.")

def get_storefront_menu() -> List[Dict]:
    return [{
        'id': p.id,
        'description': p.description,
        'price': p.price,
        'content_type': p.content_type,
        'delivery_method': p.delivery_method,
        'tags': p.tags
    } for p in PRODUCT_CATALOG]

__all__ = [
    'Product',
    'detect_premium_request',
    'handle_premium_request',
    'purchase_product',
    'get_storefront_menu'
]

