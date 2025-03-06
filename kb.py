from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton


def get_main_menu_keyboard() -> InlineKeyboardMarkup:
    """Создать главное меню на русском языке"""
    buttons = [
        [
            InlineKeyboardButton(
                text="🔄 Новый перенос стиля", callback_data="action:new_transfer"
            )
        ],
        [InlineKeyboardButton(text="⚙️ Настройки", callback_data="action:settings")],
        [InlineKeyboardButton(text="❓ Помощь", callback_data="action:help")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def get_settings_keyboard() -> InlineKeyboardMarkup:
    """Создать меню настроек на русском языке"""
    buttons = [
        # Регулировка guidance
        [
            InlineKeyboardButton(
                text="⬇️ Guidance", callback_data="setting:guidance_down"
            ),
            InlineKeyboardButton(
                text="Guidance ⬆️", callback_data="setting:guidance_up"
            ),
        ],
        # Регулировка conditioning
        [
            InlineKeyboardButton(
                text="⬇️ Condit-ng", callback_data="setting:conditioning_down"
            ),
            InlineKeyboardButton(
                text="Condit-ng ⬆️", callback_data="setting:conditioning_up"
            ),
        ],
        # Шаги
        [
            InlineKeyboardButton(text="⬇️ Steps", callback_data="setting:steps_down"),
            InlineKeyboardButton(text="Steps ⬆️", callback_data="setting:steps_up"),
        ],
        # IP Адаптер
        [
            InlineKeyboardButton(text="⬇️ IP Адаптер", callback_data="setting:ip_down"),
            InlineKeyboardButton(text="IP Адаптер ⬆️", callback_data="setting:ip_up"),
        ],
        # Готово
        [
            InlineKeyboardButton(text="✅ Готово", callback_data="action:main_menu"),
        ],
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def get_confirmation_keyboard() -> InlineKeyboardMarkup:
    """Создать клавиатуру для подтверждения на русском языке"""
    buttons = [
        [
            InlineKeyboardButton(text="✅ Да", callback_data="confirm:yes"),
            InlineKeyboardButton(text="❌ Нет", callback_data="confirm:no"),
        ]
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)
