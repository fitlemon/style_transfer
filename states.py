from aiogram.fsm.state import State, StatesGroup


class StyleTransferStates(StatesGroup):
    """States for the style transfer process"""

    waiting_for_style_photo = State()  # First photo - reference style
    waiting_for_content_photo = State()  # Second photo - content to apply style to
    waiting_for_confirmation = State()  # For future confirmation dialogs
    processing = State()  # During processing
