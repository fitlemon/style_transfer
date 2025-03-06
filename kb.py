from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton


def get_main_menu_keyboard() -> InlineKeyboardMarkup:
    """Create main menu keyboard"""
    buttons = [
        [
            InlineKeyboardButton(
                text="🔄 New Style Transfer", callback_data="action:new_transfer"
            )
        ],
        [InlineKeyboardButton(text="⚙️ Settings", callback_data="action:settings")],
        [InlineKeyboardButton(text="❓ Help", callback_data="action:help")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def get_settings_keyboard() -> InlineKeyboardMarkup:
    """Create keyboard for settings menu"""
    buttons = [
        # Guidance scale controls
        [
            InlineKeyboardButton(
                text="⬇️ Guidance", callback_data="setting:guidance_down"
            ),
            InlineKeyboardButton(
                text="Guidance ⬆️", callback_data="setting:guidance_up"
            ),
        ],
        # Conditioning scale controls
        [
            InlineKeyboardButton(
                text="⬇️ Conditioning", callback_data="setting:conditioning_down"
            ),
            InlineKeyboardButton(
                text="Conditioning ⬆️", callback_data="setting:conditioning_up"
            ),
        ],
        # Inference steps controls
        [
            InlineKeyboardButton(text="⬇️ Steps", callback_data="setting:steps_down"),
            InlineKeyboardButton(text="Steps ⬆️", callback_data="setting:steps_up"),
        ],
        # IP Adapter scale controls
        [
            InlineKeyboardButton(text="⬇️ IP Adapter", callback_data="setting:ip_down"),
            InlineKeyboardButton(text="IP Adapter ⬆️", callback_data="setting:ip_up"),
        ],
        # Done button
        [
            InlineKeyboardButton(text="✅ Done", callback_data="action:main_menu"),
        ],
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def get_confirmation_keyboard() -> InlineKeyboardMarkup:
    """Create keyboard for confirming actions"""
    buttons = [
        [
            InlineKeyboardButton(text="✅ Yes", callback_data="confirm:yes"),
            InlineKeyboardButton(text="❌ No", callback_data="confirm:no"),
        ]
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)
