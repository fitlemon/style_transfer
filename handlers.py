import os
import tempfile
import logging
import asyncio
from PIL import Image
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Any

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

# Создаем роутер
router = Router()

# Создаем глобальную очередь задач и счетчик
task_queue = deque()
queue_processing = False
last_position = 0
user_positions = {}  # Словарь для хранения позиций пользователей в очереди

# Блокировка для работы с очередью
queue_lock = asyncio.Lock()

MAX_QUEUE_SIZE = 5


@dataclass
class StyleTransferTask:
    """Класс для хранения задачи переноса стиля"""

    user_id: int
    message: Message
    style_model: Any
    style_photo_path: str
    content_photo_path: str
    settings: Dict[str, float]
    position: int
    added_time: float
    state: FSMContext


async def process_queue():
    """Асинхронный обработчик очереди"""
    global queue_processing, task_queue, user_positions
    if len(task_queue) >= MAX_QUEUE_SIZE:
        await message.answer(
            "⚠️ Очередь переполнена. Пожалуйста, попробуйте позже, когда текущие задачи будут обработаны.",
        )
        return
    if queue_processing:
        return

    queue_processing = True

    try:
        while task_queue:
            # Извлекаем задачу из очереди
            async with queue_lock:
                task = task_queue.popleft()

            # Получаем данные задачи
            message = task.message
            style_model = task.style_model
            style_photo_path = task.style_photo_path
            content_photo_path = task.content_photo_path
            settings = task.settings
            user_id = task.user_id
            state = task.state

            # Удаляем пользователя из словаря позиций
            if user_id in user_positions:
                del user_positions[user_id]

            # Обновляем позиции оставшихся пользователей
            async with queue_lock:
                for uid in user_positions:
                    if user_positions[uid] > task.position:
                        user_positions[uid] -= 1
                        # Отправляем обновление позиции в очереди
                        try:
                            await message.bot.send_message(
                                uid,
                                f"📊 Ваша позиция в очереди обновлена: {user_positions[uid]}",
                            )
                        except Exception as e:
                            logger.error(
                                f"Не удалось отправить обновление позиции: {e}"
                            )

            # Отправляем уведомление пользователю о начале обработки
            await message.answer(
                f"🚀 Начинаю обработку вашего запроса! Идёт подготовка изображений..."
            )

            try:
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

                # Добавляем сообщение о процессе генерации с эмодзи
                await message.answer(
                    "🎨 Идёт генерация стилизованного изображения... ⚡️🖌️\n\n"
                    "Это может занять до минуты, пожалуйста, подождите."
                )

                # Создаем асинхронную задачу для генерации
                # Используем обертку для запуска синхронной функции в отдельном потоке
                def run_generation():
                    return style_model.generate(
                        prompt=prompt,
                        base_img=content_img,
                        ip_adapter_img=style_img,
                        canny_img=canny_img,
                        guidance_scale=settings["guidance_scale"],
                        conditioning_scale=settings["conditioning_scale"],
                        inference_steps=settings["inference_steps"],
                        num_images=1,
                    )

                # Запускаем генерацию в отдельном потоке, чтобы не блокировать бот
                loop = asyncio.get_event_loop()
                gen_images = await loop.run_in_executor(None, run_generation)

                # Сообщаем об окончании генерации
                await message.answer("✅ Генерация завершена! Отправляю результат...")

                # Сохраняем результат
                generated_file_path = content_photo_path.replace(".jpg", "_gen.jpg")
                gen_images[0].save(generated_file_path)

                # Клавиатура с опциями после генерации
                result_kb = InlineKeyboardMarkup(
                    inline_keyboard=[
                        [
                            InlineKeyboardButton(
                                text="🔄 Новый перенос стиля",
                                callback_data="action:new_transfer",
                            )
                        ],
                        [
                            InlineKeyboardButton(
                                text="⚙️ Настройки", callback_data="action:settings"
                            )
                        ],
                        [
                            InlineKeyboardButton(
                                text="📋 Главное меню", callback_data="action:main_menu"
                            )
                        ],
                    ]
                )

                # Отправляем изображение
                await message.answer_photo(
                    FSInputFile(generated_file_path),
                    caption=(
                        "✨ Вот ваше стилизованное изображение! ✨\n\n<b>Использованные настройки:</b>\n"
                        f"• Guidance Scale: {settings['guidance_scale']}\n"
                        f"• Conditioning Scale: {settings['conditioning_scale']}\n"
                        f"• Inference Steps: {settings['inference_steps']}\n"
                        f"• IP Adapter Scale: {settings['ip_adapter_scale']}"
                    ),
                    reply_markup=result_kb,
                )

                # Удаляем временные файлы
                os.remove(style_photo_path)
                os.remove(content_photo_path)
                os.remove(generated_file_path)

                # Освобождаем память от промежуточных данных
                del style_img, content_img, canny_img, gen_images
                import gc

                gc.collect()

                # Если используется GPU, очищаем его память
                try:
                    import torch

                    torch.cuda.empty_cache()
                except ImportError:
                    pass

                # Temporarily reduce IP-Adapter memory footprint
                if hasattr(style_model.pipe, "ip_adapter"):
                    style_model.pipe.ip_adapter = None
                    torch.cuda.empty_cache()

                # Restore IP-Adapter before next task
                style_model.pipe.load_ip_adapter(
                    "h94/IP-Adapter",
                    subfolder="models",
                    weight_name="ip-adapter_sd15.bin",
                )
                # Сбрасываем состояние
                await state.clear()

            except Exception as e:
                logger.error(f"Ошибка при обработке изображения: {e}")

                error_kb = InlineKeyboardMarkup(
                    inline_keyboard=[
                        [
                            InlineKeyboardButton(
                                text="🔄 Попробовать снова",
                                callback_data="action:new_transfer",
                            )
                        ],
                        [
                            InlineKeyboardButton(
                                text="📋 Главное меню", callback_data="action:main_menu"
                            )
                        ],
                    ]
                )

                await message.answer(
                    f"❌ Ошибка при обработке вашего изображения: {str(e)}\n\nПопробуйте снова.",
                    reply_markup=error_kb,
                )

                # Удаляем временные файлы в случае ошибки
                if os.path.exists(style_photo_path):
                    os.remove(style_photo_path)
                if os.path.exists(content_photo_path):
                    os.remove(content_photo_path)

                # Сбрасываем состояние
                await state.clear()

            # Делаем небольшую паузу между задачами
            await asyncio.sleep(1)

            try:
                style_model.cleanup()
            except Exception as e:
                logger.error(f"Error during model cleanup: {e}")

    except Exception as e:
        logger.error(f"Ошибка в обработчике очереди: {e}")
    finally:
        queue_processing = False
        # Если в очереди остались задачи, запускаем обработчик снова
        if task_queue:
            asyncio.create_task(process_queue())


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


@router.message(Command("queue"))
async def cmd_queue(message: Message):
    """Показать текущую очередь и позицию пользователя"""
    user_id = message.from_user.id

    if len(task_queue) == 0:
        await message.answer(
            "🟢 В данный момент очередь пуста, вы можете сразу начать перенос стиля!",
            reply_markup=get_main_menu_keyboard(),
        )
        return

    if user_id in user_positions:
        position = user_positions[user_id]
        await message.answer(
            f"📊 Ваша текущая позиция в очереди: {position} из {len(task_queue)}\n\n"
            f"Примерное время ожидания: {position * 1.5} минут",
            reply_markup=get_main_menu_keyboard(),
        )
    else:
        await message.answer(
            f"🔍 У вас нет активных задач в очереди.\n\n"
            f"Текущая длина очереди: {len(task_queue)} запросов",
            reply_markup=get_main_menu_keyboard(),
        )


@router.message(Command("settings"))
async def cmd_settings(message: Message, state: FSMContext):
    """Обработчик команды /settings"""
    # Получаем текущие настройки или используем значения по умолчанию
    data = await state.get_data()
    settings = {
        "guidance_scale": data.get("guidance_scale", 6.0),
        "conditioning_scale": data.get("conditioning_scale", 0.7),
        "inference_steps": data.get("inference_steps", 20),
        "ip_adapter_scale": data.get("ip_adapter_scale", 0.5),
    }

    await message.answer(
        f"<b>Текущие настройки:</b>\n\n"
        f"Guidance Scale: {settings['guidance_scale']}\n"
        f"Conditioning Scale: {settings['conditioning_scale']}\n"
        f"Inference Steps: {settings['inference_steps']}\n"
        f"IP Adapter Scale: {settings['ip_adapter_scale']}\n\n"
        f"Выберите параметр для изменения:",
        reply_markup=get_settings_keyboard(),
    )


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext):
    """Отменить текущую операцию и сбросить состояние"""
    current_state = await state.get_state()
    user_id = message.from_user.id

    if current_state is None:
        # Проверяем, есть ли пользователь в очереди
        if user_id in user_positions:
            position = user_positions[user_id]
            # Удаляем задачу из очереди
            async with queue_lock:
                # Находим задачу в очереди
                task_to_remove = None
                for task in task_queue:
                    if task.user_id == user_id:
                        task_to_remove = task
                        break

                if task_to_remove:
                    task_queue.remove(task_to_remove)
                    del user_positions[user_id]

                    # Обновляем позиции оставшихся пользователей
                    for uid in user_positions:
                        if user_positions[uid] > position:
                            user_positions[uid] -= 1
                            # Отправляем обновление позиции в очереди
                            try:
                                await message.bot.send_message(
                                    uid,
                                    f"📊 Ваша позиция в очереди обновлена: {user_positions[uid]}",
                                )
                            except Exception as e:
                                logger.error(
                                    f"Не удалось отправить обновление позиции: {e}"
                                )

                    await message.answer(
                        "Ваш запрос в очереди отменен. Вы можете начать новую задачу из меню ниже.",
                        reply_markup=get_main_menu_keyboard(),
                    )
                    return

        await message.answer(
            "Нет активных операций для отмены.", reply_markup=get_main_menu_keyboard()
        )
        return

    # Удаляем временные файлы, если они есть
    data = await state.get_data()
    style_photo_path = data.get("style_photo_path")
    content_photo_path = data.get("content_photo_path")

    if style_photo_path and os.path.exists(style_photo_path):
        os.remove(style_photo_path)

    if content_photo_path and os.path.exists(content_photo_path):
        os.remove(content_photo_path)

    # Сбрасываем состояние
    await state.clear()

    # Если пользователь был в очереди, удаляем его
    if user_id in user_positions:
        position = user_positions[user_id]
        # Удаляем задачу из очереди
        async with queue_lock:
            # Находим задачу в очереди
            task_to_remove = None
            for task in task_queue:
                if task.user_id == user_id:
                    task_to_remove = task
                    break

            if task_to_remove:
                task_queue.remove(task_to_remove)
                del user_positions[user_id]

                # Обновляем позиции оставшихся пользователей
                for uid in user_positions:
                    if user_positions[uid] > position:
                        user_positions[uid] -= 1
                        # Отправляем обновление позиции в очереди
                        try:
                            await message.bot.send_message(
                                uid,
                                f"📊 Ваша позиция в очереди обновлена: {user_positions[uid]}",
                            )
                        except Exception as e:
                            logger.error(
                                f"Не удалось отправить обновление позиции: {e}"
                            )

    await message.answer(
        "Операция отменена. Вы можете начать новую задачу из меню ниже.",
        reply_markup=get_main_menu_keyboard(),
    )


@router.callback_query(F.data == "action:new_transfer")
async def start_new_transfer(callback: CallbackQuery, state: FSMContext):
    """Начать новый процесс переноса стиля"""
    await callback.answer()

    # Устанавливаем состояние ожидания стилевого фото
    await state.set_state(StyleTransferStates.waiting_for_style_photo)

    await callback.message.answer(
        "Начнём новый перенос стиля! 🎨\n\n"
        "Сначала отправьте мне фото-референс стиля — по нему будет определяться применяемый стиль."
    )


@router.message(StyleTransferStates.waiting_for_style_photo, F.photo)
async def on_style_photo(message: Message, state: FSMContext):
    """Обработка полученного фотографии-референса стиля"""
    # Загружаем фото
    await message.answer("Стилевое фото получено! ✅")

    # Берём самое высокое разрешение фото
    file_id = message.photo[-1].file_id
    file = await message.bot.get_file(file_id)
    file_path = file.file_path

    # Сохраняем фото во временный файл
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    await message.bot.download_file(file_path, temp_file.name)
    temp_file.close()

    # Сохраняем путь к файлу в состоянии
    await state.update_data(style_photo_path=temp_file.name)

    # Устанавливаем состояние ожидания фото-контента
    await state.set_state(StyleTransferStates.waiting_for_content_photo)

    # Клавиатура с кнопкой отмены
    cancel_kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="❌ Отмена", callback_data="action:cancel")]
        ]
    )

    await message.answer(
        "Отлично! Теперь отправьте фото, к которому вы хотите применить этот стиль.",
        reply_markup=cancel_kb,
    )


@router.message(StyleTransferStates.waiting_for_content_photo, F.photo)
async def on_content_photo(message: Message, state: FSMContext, style_model):
    """Обработка полученного фото-контента"""
    global last_position, user_positions

    user_id = message.from_user.id

    # Загружаем фото
    await message.answer("Фото-контент получен! Добавляю в очередь на обработку... ⏳")

    # Берём самое высокое разрешение фото
    file_id = message.photo[-1].file_id
    file = await message.bot.get_file(file_id)
    file_path = file.file_path

    # Сохраняем фото во временный файл
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    await message.bot.download_file(file_path, temp_file.name)
    temp_file.close()

    # Сохраняем путь к файлу в состоянии
    await state.update_data(content_photo_path=temp_file.name)

    # Получаем данные из состояния для обоих изображений
    data = await state.get_data()
    style_photo_path = data.get("style_photo_path")
    content_photo_path = temp_file.name

    # Устанавливаем состояние "обработка"
    await state.set_state(StyleTransferStates.processing)

    # Получаем текущие настройки
    settings = {
        "guidance_scale": data.get("guidance_scale", 6.0),
        "conditioning_scale": data.get("conditioning_scale", 0.7),
        "inference_steps": data.get("inference_steps", 20),
        "ip_adapter_scale": data.get("ip_adapter_scale", 0.5),
    }

    # Обновляем счетчик позиции
    async with queue_lock:
        last_position += 1
        current_position = last_position
        user_positions[user_id] = current_position

        # Создаем задачу и добавляем её в очередь
        task = StyleTransferTask(
            user_id=user_id,
            message=message,
            style_model=style_model,
            style_photo_path=style_photo_path,
            content_photo_path=content_photo_path,
            settings=settings,
            position=current_position,
            added_time=time.time(),
            state=state,
        )

        task_queue.append(task)

    # Оповещаем пользователя о его позиции в очереди
    queue_length = len(task_queue)
    estimated_time = queue_length * 1.5  # примерно 1.5 минуты на задачу

    await message.answer(
        f"📋 Ваш запрос добавлен в очередь!\n\n"
        f"Позиция в очереди: {current_position}\n"
        f"Всего запросов в очереди: {queue_length}\n"
        f"Примерное время ожидания: {estimated_time:.1f} мин\n\n"
        f"Мы отправим уведомление, когда начнется обработка вашего запроса.\n"
        f"Вы можете узнать текущую позицию с помощью команды /queue",
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="❌ Отменить запрос", callback_data="action:cancel"
                    )
                ]
            ]
        ),
    )

    # Запускаем обработчик очереди, если он еще не запущен
    if not queue_processing and queue_length == 1:
        asyncio.create_task(process_queue())


@router.message(F.photo)
async def photo_without_state(message: Message):
    """Обработка фотографий без установленного состояния"""
    options_kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🔄 Начать перенос стиля", callback_data="action:new_transfer"
                )
            ],
            [
                InlineKeyboardButton(
                    text="📋 Главное меню", callback_data="action:main_menu"
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
    """Показать главное меню"""
    await callback.answer()

    # Проверяем, находится ли пользователь в очереди
    user_id = callback.from_user.id
    if user_id in user_positions:
        # Спрашиваем, действительно ли он хочет выйти и отменить запрос
        confirm_kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="✅ Да, отменить запрос",
                        callback_data="confirm:cancel_queue",
                    )
                ],
                [
                    InlineKeyboardButton(
                        text="❌ Нет, оставить в очереди",
                        callback_data="confirm:keep_queue",
                    )
                ],
            ]
        )

        await callback.message.answer(
            "⚠️ У вас есть активный запрос в очереди. Вы действительно хотите отменить его?",
            reply_markup=confirm_kb,
        )
        return

    await state.clear()  # Сбрасываем текущее состояние

    await callback.message.answer(
        "📋 <b>Главное меню</b>\n\nВыберите действие из списка ниже:",
        reply_markup=get_main_menu_keyboard(),
    )


@router.callback_query(F.data == "confirm:cancel_queue")
async def confirm_cancel_queue(callback: CallbackQuery, state: FSMContext):
    """Подтверждение отмены запроса в очереди"""
    user_id = callback.from_user.id

    if user_id in user_positions:
        position = user_positions[user_id]
        # Удаляем задачу из очереди
        async with queue_lock:
            # Находим задачу в очереди
            task_to_remove = None
            for task in task_queue:
                if task.user_id == user_id:
                    task_to_remove = task

                    # Удаляем временные файлы
                    if os.path.exists(task.style_photo_path):
                        os.remove(task.style_photo_path)
                    if os.path.exists(task.content_photo_path):
                        os.remove(task.content_photo_path)
                    break

            if task_to_remove:
                task_queue.remove(task_to_remove)
                del user_positions[user_id]

                # Обновляем позиции оставшихся пользователей
                for uid in user_positions:
                    if user_positions[uid] > position:
                        user_positions[uid] -= 1
                        # Отправляем обновление позиции в очереди
                        try:
                            await callback.bot.send_message(
                                uid,
                                f"📊 Ваша позиция в очереди обновлена: {user_positions[uid]}",
                            )
                        except Exception as e:
                            logger.error(
                                f"Не удалось отправить обновление позиции: {e}"
                            )

    await callback.answer("Запрос отменен")
    await state.clear()

    await callback.message.answer(
        "Ваш запрос отменен. Вы можете начать новую задачу из меню ниже.",
        reply_markup=get_main_menu_keyboard(),
    )


@router.callback_query(F.data == "confirm:keep_queue")
async def confirm_keep_queue(callback: CallbackQuery):
    """Подтверждение сохранения запроса в очереди"""
    await callback.answer("Запрос оставлен в очереди")

    user_id = callback.from_user.id
    if user_id in user_positions:
        position = user_positions[user_id]
        queue_length = len(task_queue)
        estimated_time = position * 1.5  # примерно 1.5 минуты на задачу

        await callback.message.answer(
            f"📋 Ваш запрос остается в очереди!\n\n"
            f"Текущая позиция: {position}\n"
            f"Всего запросов в очереди: {queue_length}\n"
            f"Примерное время ожидания: {estimated_time:.1f} мин\n\n"
            f"Используйте команду /queue, чтобы проверить статус.",
            reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text="❌ Отменить запрос", callback_data="action:cancel"
                        )
                    ]
                ]
            ),
        )


@router.callback_query(F.data == "action:settings")
async def show_settings(callback: CallbackQuery, state: FSMContext):
    """Показать меню настроек через callback"""
    await callback.answer()

    # Получаем текущие настройки или используем по умолчанию
    data = await state.get_data()
    settings = {
        "guidance_scale": data.get("guidance_scale", 6.0),
        "conditioning_scale": data.get("conditioning_scale", 0.7),
        "inference_steps": data.get("inference_steps", 20),
        "ip_adapter_scale": data.get("ip_adapter_scale", 0.5),
    }

    await callback.message.answer(
        f"<b>Текущие настройки:</b>\n\n"
        f"Guidance Scale: {settings['guidance_scale']}\n"
        f"Conditioning Scale: {settings['conditioning_scale']}\n"
        f"Inference Steps: {settings['inference_steps']}\n"
        f"IP Adapter Scale: {settings['ip_adapter_scale']}\n\n"
        f"Выберите параметр для изменения:",
        reply_markup=get_settings_keyboard(),
    )


@router.callback_query(F.data == "action:help")
async def show_help(callback: CallbackQuery):
    """Показать справку через callback"""
    await callback.answer()

    help_text = (
        "🖼 <b>Справка по StyleTransfer Bot</b>\n\n"
        "Этот бот позволяет применить стиль одного изображения к другому.\n\n"
        "<b>Команды:</b>\n"
        "/start - Запустить бота и показать главное меню\n"
        "/help - Показать это сообщение справки\n"
        "/settings - Настроить параметры style transfer\n"
        "/queue - Проверить вашу позицию в очереди\n"
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

    await callback.message.answer(help_text, reply_markup=get_main_menu_keyboard())


@router.callback_query(F.data.startswith("setting:"))
async def settings_callback(callback: CallbackQuery, state: FSMContext):
    """Обработка изменений настроек"""
    action = callback.data.split(":")[1]
    data = await state.get_data()

    # Получаем текущие настройки или используем по умолчанию
    guidance_scale = data.get("guidance_scale", 6.0)
    conditioning_scale = data.get("conditioning_scale", 0.7)
    inference_steps = data.get("inference_steps", 20)
    ip_adapter_scale = data.get("ip_adapter_scale", 0.5)

    # Обновляем настройки в зависимости от действия
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

    # Обновляем состояние с новыми настройками
    await state.update_data(
        guidance_scale=guidance_scale,
        conditioning_scale=conditioning_scale,
        inference_steps=inference_steps,
        ip_adapter_scale=ip_adapter_scale,
    )

    # Обновляем сообщение с настройками
    await callback.message.edit_text(
        f"<b>Текущие настройки:</b>\n\n"
        f"Guidance Scale: {guidance_scale}\n"
        f"Conditioning Scale: {conditioning_scale}\n"
        f"Inference Steps: {inference_steps}\n"
        f"IP Adapter Scale: {ip_adapter_scale}\n\n"
        f"Выберите параметр для изменения:",
        reply_markup=get_settings_keyboard(),
    )

    await callback.answer(f"Настройки обновлены")


# Функция для вычисления примерного времени ожидания
def estimate_wait_time(position: int) -> float:
    """Вычислить примерное время ожидания в минутах"""
    # В среднем обработка одного запроса занимает около 1.5 минут
    average_task_time = 1.5  # в минутах
    return position * average_task_time


# Периодическое обновление информации о времени ожидания
async def update_queue_times():
    """Периодически обновляет информацию о времени ожидания для пользователей в очереди"""
    while True:
        await asyncio.sleep(60)  # Обновляем каждую минуту

        if not task_queue:
            continue

        try:
            async with queue_lock:
                for uid, position in user_positions.items():
                    try:
                        # Находим задачу пользователя
                        for task in task_queue:
                            if task.user_id == uid:
                                # Вычисляем, сколько времени прошло с момента добавления задачи
                                elapsed_time = (
                                    time.time() - task.added_time
                                ) / 60  # в минутах

                                # Обновляем оценку времени ожидания
                                remaining_time = max(
                                    0, estimate_wait_time(position) - elapsed_time
                                )

                                # Отправляем обновление только если осталось значительное время
                                if remaining_time > 2 and position > 1:
                                    await task.message.bot.send_message(
                                        uid,
                                        f"📊 Обновление по вашему запросу в очереди:\n\n"
                                        f"Текущая позиция: {position} из {len(task_queue)}\n"
                                        f"Примерное оставшееся время: {remaining_time:.1f} мин",
                                    )
                                break
                    except Exception as e:
                        logger.error(f"Ошибка при обновлении времени ожидания: {e}")
        except Exception as e:
            logger.error(f"Ошибка в задаче обновления очереди: {e}")


# Запускаем задачу обновления времени ожидания при старте бота
async def start_queue_updates():
    """Запустить периодические обновления очереди"""
    asyncio.create_task(update_queue_times())


# Функция для получения статистики по очереди
@router.message(Command("stats"))
async def cmd_stats(message: Message):
    """Показать статистику очереди и текущие задачи (только для админов)"""
    # Здесь можно добавить проверку на администратора
    # if message.from_user.id != ADMIN_ID:
    #    await message.answer("У вас нет прав для выполнения этой команды.")
    #    return

    stats_text = (
        f"📊 <b>Статистика очереди:</b>\n\n"
        f"Всего задач в очереди: {len(task_queue)}\n"
        f"Активных пользователей в очереди: {len(user_positions)}\n\n"
    )

    if task_queue:
        stats_text += "<b>Текущие задачи:</b>\n"
        for i, task in enumerate(task_queue, 1):
            elapsed = (time.time() - task.added_time) / 60
            stats_text += f"{i}. ID: {task.user_id}, ждёт {elapsed:.1f} мин\n"
    else:
        stats_text += "Очередь пуста."

    await message.answer(stats_text)


# Обработчик для отмены задачи через кнопку
@router.callback_query(F.data == "action:cancel")
async def cancel_task_callback(callback: CallbackQuery, state: FSMContext):
    """Отменить текущую задачу через callback"""
    current_state = await state.get_state()
    user_id = callback.from_user.id

    if current_state is not None:
        # Удаляем временные файлы, если они есть
        data = await state.get_data()
        style_photo_path = data.get("style_photo_path")
        content_photo_path = data.get("content_photo_path")

        if style_photo_path and os.path.exists(style_photo_path):
            os.remove(style_photo_path)

        if content_photo_path and os.path.exists(content_photo_path):
            os.remove(content_photo_path)

        # Сбрасываем состояние
        await state.clear()

    # Проверяем, есть ли задача в очереди
    if user_id in user_positions:
        position = user_positions[user_id]
        async with queue_lock:
            # Находим задачу в очереди
            task_to_remove = None
            for task in task_queue:
                if task.user_id == user_id:
                    task_to_remove = task
                    break

            if task_to_remove:
                # Удаляем временные файлы задачи
                if os.path.exists(task_to_remove.style_photo_path):
                    os.remove(task_to_remove.style_photo_path)
                if os.path.exists(task_to_remove.content_photo_path):
                    os.remove(task_to_remove.content_photo_path)

                # Удаляем задачу из очереди
                task_queue.remove(task_to_remove)
                del user_positions[user_id]

                # Обновляем позиции оставшихся пользователей
                for uid in user_positions:
                    if user_positions[uid] > position:
                        user_positions[uid] -= 1
                        # Отправляем обновление позиции в очереди
                        try:
                            await callback.bot.send_message(
                                uid,
                                f"📊 Ваша позиция в очереди обновлена: {user_positions[uid]}",
                            )
                        except Exception as e:
                            logger.error(
                                f"Не удалось отправить обновление позиции: {e}"
                            )

    await callback.answer("Задача отменена")
    await callback.message.answer(
        "Операция отменена. Вы можете начать новую задачу из меню ниже.",
        reply_markup=get_main_menu_keyboard(),
    )
