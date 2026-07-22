import logging
import os
from dotenv import load_dotenv
from inference import ProductivityPredictor
from telegram import ReplyKeyboardRemove, Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

load_dotenv(".env")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

predictor = ProductivityPredictor(model_dir="models")
FEATURES = predictor.feature_names

ASK_FEATURE = 0


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Starts conversation and prompts for the first feature."""
    context.user_data.clear()
    context.user_data["current_step"] = 0

    first_feature = FEATURES[0]
    await update.message.reply_text(
        "👋 **Welcome to the Productivity Predictor Bot!**\n\n"
        "I will ask you a few questions about your day to predict your"
        " productivity score.\n\n"
        f"**Question 1/{len(FEATURES)}:** Enter your value for **{first_feature}**:",
        parse_mode="Markdown",
    )
    return ASK_FEATURE


async def handle_feature_input(
        update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
    """Collects input, validates numeric value, and moves to the next question."""
    user_text = update.message.text
    step = context.user_data.get("current_step", 0)
    current_feature = FEATURES[step]

    try:
        value = float(user_text)
    except ValueError:
        await update.message.reply_text(
            f"⚠️ Please enter a valid number for **{current_feature}**.",
            parse_mode="Markdown",
        )
        return ASK_FEATURE

    context.user_data[current_feature] = value
    step += 1
    context.user_data["current_step"] = step

    if step < len(FEATURES):
        next_feature = FEATURES[step]
        await update.message.reply_text(
            f"**Question {step + 1}/{len(FEATURES)}:** Enter your value for"
            f" **{next_feature}**:",
            parse_mode="Markdown",
        )
        return ASK_FEATURE

    await update.message.reply_text(
        "🤖 Calculating your productivity score..."
    )

    user_inputs = {feat: context.user_data[feat] for feat in FEATURES}
    predicted_score = predictor.predict(user_inputs)

    await update.message.reply_text(
        f"🎯 **Your Predicted Productivity Score:** `{predicted_score:.1f} /"
        " 100`\n\n"
        "Type /start to run another prediction!",
        parse_mode="Markdown",
    )
    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels the current active session."""
    await update.message.reply_text(
        "Prediction process cancelled. Type /start whenever you're ready again!",
        reply_markup=ReplyKeyboardRemove(),
    )
    return ConversationHandler.END


def main():
    if not TELEGRAM_TOKEN:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN missing! Make sure it is set in your .env file."
        )

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            ASK_FEATURE: [
                MessageHandler(
                    filters.TEXT & ~filters.COMMAND, handle_feature_input
                )
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    app.add_handler(conv_handler)

    print("Bot is running... Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()