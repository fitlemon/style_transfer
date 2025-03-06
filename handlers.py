import os
import tempfile
import logging
from PIL import Image

from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    FSInputFile,
)
from aiogram.fsm.context import FSMContext

from states import StyleTransferStates
from kb import get_settings_keyboard, get_confirmation_keyboard, get_main_menu_keyboard

logger = logging.getLogger(__name__)

# Create router
router = Router()


@router.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext):
    """Обработчик команды /start"""
    # Очищаем предыдущее состояние
    await state.clear()

    await message.answer(
        "Добро пожаловать в StyleTransfer Bot! 🎨\n\n"
        "Этот бот позволяет применить стиль одного изображения к другому.\n\n"
        "Выберите действие из меню ниже:",
        reply_markup=get_main_menu_keyboard(),
    )


@router.message(Command("help"))
async def cmd_help(message: Message):
    """Обработчик команды /help"""
    help_text = (
        "🖼 <b>Справка по StyleTransfer Bot</b>\n\n"
        "Этот бот позволяет применить стиль одного изображения к другому.\n\n"
        "<b>Команды:</b>\n"
        "/start - Запустить бота и показать главное меню\n"
        "/help - Показать это сообщение справки\n"
        "/settings - Настроить параметры style transfer\n"
        "/cancel - Отменить текущую операцию\n\n"
        "<b>Как использовать:</b>\n"
        "1. Выберите «🔄 Новый перенос стиля» в меню\n"
        "2. Отправьте фото-референс стиля\n"
        "3. Отправьте фото, к которому хотите применить выбранный стиль\n"
        "4. Дождитесь своей очереди и завершения обработки (это может занять некоторое время)\n"
        "5. Получите стилизованное изображение\n\n"
        "<b>Очередь:</b>\n"
        "Если несколько пользователей запрашивают перенос стиля одновременно, "
        "запросы будут обрабатываться в порядке очереди. Вы будете уведомлены "
        "о вашей позиции в очереди и когда начнется обработка вашего запроса.\n\n"
        "<b>Настройки:</b>\n"
        "- Guidance Scale: Controls how closely the image follows the prompt\n"
        "- Conditioning Scale: Controls the influence of the control image\n"
        "- Inference Steps: Controls the quality (more steps = better quality but slower)\n"
        "- IP Adapter Scale: Controls the strength of the style transfer"
    )
    await message.answer(help_text, reply_markup=get_main_menu_keyboard())


@router.message(Command("settings"))
async def cmd_settings(message: Message, state: FSMContext):
    """Handle the /settings command"""
    # Get current settings or use defaults
    data = await state.get_data()
    settings = {
        "guidance_scale": data.get("guidance_scale", 6.0),
        "conditioning_scale": data.get("conditioning_scale", 0.7),
        "inference_steps": data.get("inference_steps", 20),
        "ip_adapter_scale": data.get("ip_adapter_scale", 0.5),
    }

    await message.answer(
        f"<b>Current Settings:</b>\n\n"
        f"Guidance Scale: {settings['guidance_scale']}\n"
        f"Conditioning Scale: {settings['conditioning_scale']}\n"
        f"Inference Steps: {settings['inference_steps']}\n"
        f"IP Adapter Scale: {settings['ip_adapter_scale']}\n\n"
        f"Select a parameter to change:",
        reply_markup=get_settings_keyboard(),
    )


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext):
    """Cancel the current operation and reset state"""
    current_state = await state.get_state()

    if current_state is None:
        await message.answer(
            "No active operations to cancel.", reply_markup=get_main_menu_keyboard()
        )
        return

    # Clean up any temporary files if they exist
    data = await state.get_data()
    style_photo_path = data.get("style_photo_path")
    content_photo_path = data.get("content_photo_path")

    if style_photo_path and os.path.exists(style_photo_path):
        os.remove(style_photo_path)

    if content_photo_path and os.path.exists(content_photo_path):
        os.remove(content_photo_path)

    # Clear the state
    await state.clear()

    await message.answer(
        "Operation canceled. You can start a new task from the menu below.",
        reply_markup=get_main_menu_keyboard(),
    )


@router.callback_query(F.data == "action:new_transfer")
async def start_new_transfer(callback: CallbackQuery, state: FSMContext):
    """Start a new style transfer process"""
    await callback.answer()

    # Set state to waiting for style photo
    await state.set_state(StyleTransferStates.waiting_for_style_photo)

    await callback.message.answer(
        "Начнём новый перенос стиля! 🎨\n\n"
        "Сначала отправьте мне фото-референс стиля — по нему будет определяться применяемый стиль."
    )


@router.message(StyleTransferStates.waiting_for_style_photo, F.photo)
async def on_style_photo(message: Message, state: FSMContext):
    """Handle incoming style reference photo"""
    # Download the photo
    await message.answer("Стилевое фото получено! ✅")

    # Get the highest resolution photo
    file_id = message.photo[-1].file_id
    file = await message.bot.get_file(file_id)
    file_path = file.file_path

    # Download the file to a temporary location
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    await message.bot.download_file(file_path, temp_file.name)
    temp_file.close()

    # Save the file path in state
    await state.update_data(style_photo_path=temp_file.name)

    # Set state to waiting for content photo
    await state.set_state(StyleTransferStates.waiting_for_content_photo)

    # Create a keyboard with a cancel option
    cancel_kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="❌ Cancel", callback_data="action:cancel")]
        ]
    )

    await message.answer(
        "Отлично! Теперь отправьте фото, к которому вы хотите применить этот стиль.",
        reply_markup=cancel_kb,
    )


@router.message(StyleTransferStates.waiting_for_content_photo, F.photo)
async def on_content_photo(message: Message, state: FSMContext, style_model):
    """Handle incoming content photo"""
    # Download the photo
    # Загружаем фото
    await message.answer("Фото-контент получен! Добавляю в очередь на обработку... ⏳")

    # Get the highest resolution photo
    file_id = message.photo[-1].file_id
    file = await message.bot.get_file(file_id)
    file_path = file.file_path

    # Download the file to a temporary location
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    await message.bot.download_file(file_path, temp_file.name)
    temp_file.close()

    # Save the file path in state
    await state.update_data(content_photo_path=temp_file.name)

    # Get state data with both images
    data = await state.get_data()
    style_photo_path = data.get("style_photo_path")
    content_photo_path = temp_file.name

    # Set processing state
    await state.set_state(StyleTransferStates.processing)

    # Process the images
    try:
        # Get settings
        guidance_scale = data.get("guidance_scale", 6.0)
        conditioning_scale = data.get("conditioning_scale", 0.7)
        inference_steps = data.get("inference_steps", 20)
        ip_adapter_scale = data.get("ip_adapter_scale", 0.5)

        # Update the model's IP adapter scale if different from current
        if ip_adapter_scale != style_model.ip_adapter_scale:  # Use the class attribute
            style_model.set_ip_adapter_scale(ip_adapter_scale)  # Use the new method

        # Загружаем и подготавливаем изображения
        style_img = style_model.preprocess_image(style_photo_path)
        content_img = style_model.preprocess_image(content_photo_path)

        # Генерируем Canny изображение из content фото
        canny_img = style_model.get_canny_image(
            content_img, detect_resolution=content_img.size[1]
        )

        # Генерируем промпт, используя оба изображения
        await message.answer("🔍 Анализирую содержание и стиль изображений...")
        prompt = style_model.generate_prompt(content_img, style_img)

        # Сообщаем пользователю о начале генерации с промптом
        await message.answer(f"Переношу стиль с промптом:\n{prompt}")

        gen_images = style_model.generate(
            prompt=prompt,
            base_img=content_img,
            ip_adapter_img=style_img,
            canny_img=canny_img,
            guidance_scale=guidance_scale,
            conditioning_scale=conditioning_scale,
            inference_steps=inference_steps,
            num_images=1,
        )

        # Save the generated image
        generated_file_path = content_photo_path.replace(".jpg", "_gen.jpg")
        gen_images[0].save(generated_file_path)

        # Create keyboard for after generation
        result_kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="🔄 New Style Transfer",
                        callback_data="action:new_transfer",
                    )
                ],
                [
                    InlineKeyboardButton(
                        text="⚙️ Settings", callback_data="action:settings"
                    )
                ],
                [
                    InlineKeyboardButton(
                        text="📋 Main Menu", callback_data="action:main_menu"
                    )
                ],
            ]
        )

        # Send the generated image using FSInputFile instead of opening the file
        await message.answer_photo(
            FSInputFile(generated_file_path),
            caption=f"✨Вот стилизованная картинка! ✨\n\n<b>Settings used:</b>\n"
            f"• Guidance Scale: {guidance_scale}\n"
            f"• Conditioning Scale: {conditioning_scale}\n"
            f"• Inference Steps: {inference_steps}\n"
            f"• IP Adapter Scale: {ip_adapter_scale}",
            reply_markup=result_kb,
        )

        # Clean up temp files
        os.remove(style_photo_path)
        os.remove(content_photo_path)
        os.remove(generated_file_path)
        style_img.close()
        content_img.close()
        gen_images[0].close()

        # Reset state
        await state.clear()

    except Exception as e:
        logger.error(f"Error processing image: {e}")

        # Create menu keyboard for error recovery
        error_kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="🔄 Try Again", callback_data="action:new_transfer"
                    )
                ],
                [
                    InlineKeyboardButton(
                        text="📋 Main Menu", callback_data="action:main_menu"
                    )
                ],
            ]
        )

        await message.answer(
            f"❌ Error processing your image: {str(e)}\n\nPlease try again.",
            reply_markup=error_kb,
        )

        # Clean up temp files on error
        if os.path.exists(style_photo_path):
            os.remove(style_photo_path)
        if os.path.exists(content_photo_path):
            os.remove(content_photo_path)

        # Reset state
        await state.clear()


@router.message(F.photo)
async def photo_without_state(message: Message):
    """Handle photos when no specific state is set"""
    # Create a keyboard with options
    options_kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🔄 Начать перенос стиля", callback_data="action:new_transfer"
                )
            ],
            [
                InlineKeyboardButton(
                    text="📋 Главное меню"", callback_data="action:main_menu"
                )
            ],
        ]
    )

    await message.answer(
        "Вы отправили фото, но я пока не понимаю, что с ним делать.\n\n"
        "Чтобы начать процесс переноса стиля, воспользуйтесь кнопкой ниже:",
        reply_markup=options_kb,
    )


@router.callback_query(F.data == "action:main_menu")
async def show_main_menu(callback: CallbackQuery, state: FSMContext):
    """Show the main menu"""
    await callback.answer()
    await state.clear()  # Clear any existing state

    await callback.message.answer(
        "📋 <b>Главное меню</b>\n\nВыберите действие из списка ниже:",
        reply_markup=get_main_menu_keyboard(),
    )


@router.callback_query(F.data == "action:settings")
async def show_settings(callback: CallbackQuery, state: FSMContext):
    """Show settings menu via callback"""
    await callback.answer()

    # Get current settings or use defaults
    data = await state.get_data()
    settings = {
        "guidance_scale": data.get("guidance_scale", 6.0),
        "conditioning_scale": data.get("conditioning_scale", 0.7),
        "inference_steps": data.get("inference_steps", 20),
        "ip_adapter_scale": data.get("ip_adapter_scale", 0.5),
    }

    await callback.message.answer(
        f"<b>Current Settings:</b>\n\n"
        f"Guidance Scale: {settings['guidance_scale']}\n"
        f"Conditioning Scale: {settings['conditioning_scale']}\n"
        f"Inference Steps: {settings['inference_steps']}\n"
        f"IP Adapter Scale: {settings['ip_adapter_scale']}\n\n"
        f"Select a parameter to change:",
        reply_markup=get_settings_keyboard(),
    )


@router.callback_query(F.data == "action:cancel")
async def cancel_callback(callback: CallbackQuery, state: FSMContext):
    """Cancel the current operation via callback"""
    await callback.answer("Operation canceled")

    # Clean up any temporary files if they exist
    data = await state.get_data()
    style_photo_path = data.get("style_photo_path")
    content_photo_path = data.get("content_photo_path")

    if style_photo_path and os.path.exists(style_photo_path):
        os.remove(style_photo_path)

    if content_photo_path and os.path.exists(content_photo_path):
        os.remove(content_photo_path)

    # Clear the state
    await state.clear()

    await callback.message.answer(
        "Operation canceled. You can start a new task from the menu below.",
        reply_markup=get_main_menu_keyboard(),
    )


@router.callback_query(F.data.startswith("setting:"))
async def settings_callback(callback: CallbackQuery, state: FSMContext):
    """Handle settings callback"""
    action = callback.data.split(":")[1]
    data = await state.get_data()

    # Get current settings or use defaults
    guidance_scale = data.get("guidance_scale", 6.0)
    conditioning_scale = data.get("conditioning_scale", 0.7)
    inference_steps = data.get("inference_steps", 20)
    ip_adapter_scale = data.get("ip_adapter_scale", 0.5)

    # Update settings based on callback
    if action == "guidance_up":
        guidance_scale = min(guidance_scale + 0.5, 10.0)
    elif action == "guidance_down":
        guidance_scale = max(guidance_scale - 0.5, 1.0)
    elif action == "conditioning_up":
        conditioning_scale = min(conditioning_scale + 0.1, 1.0)
    elif action == "conditioning_down":
        conditioning_scale = max(conditioning_scale - 0.1, 0.1)
    elif action == "steps_up":
        inference_steps = min(inference_steps + 5, 50)
    elif action == "steps_down":
        inference_steps = max(inference_steps - 5, 10)
    elif action == "ip_up":
        ip_adapter_scale = min(ip_adapter_scale + 0.1, 1.0)
    elif action == "ip_down":
        ip_adapter_scale = max(ip_adapter_scale - 0.1, 0.1)
    elif action == "done":
        await callback.answer("Settings saved!")
        await callback.message.answer(
            "Settings have been saved. You can now start a new style transfer.",
            reply_markup=get_main_menu_keyboard(),
        )
        return

    # Update state with new settings
    await state.update_data(
        guidance_scale=guidance_scale,
        conditioning_scale=conditioning_scale,
        inference_steps=inference_steps,
        ip_adapter_scale=ip_adapter_scale,
    )

    # Update the settings message
    await callback.message.edit_text(
        f"<b>Current Settings:</b>\n\n"
        f"Guidance Scale: {guidance_scale}\n"
        f"Conditioning Scale: {conditioning_scale}\n"
        f"Inference Steps: {inference_steps}\n"
        f"IP Adapter Scale: {ip_adapter_scale}\n\n"
        f"Select a parameter to change:",
        reply_markup=get_settings_keyboard(),
    )

    await callback.answer(f"Settings updated")
