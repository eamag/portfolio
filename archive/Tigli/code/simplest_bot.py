from telegram.ext import Updater

updater = Updater(token='355058372:AAFKVF2NQXRvrhlHoBf0c809xeYmAyJMGBc')
dispatcher = updater.dispatcher
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


def start(bot, update):
    bot.sendMessage(chat_id=update.message.chat_id, text="Send me some twitts")


from telegram.ext import CommandHandler

start_handler = CommandHandler('start', start)
dispatcher.add_handler(start_handler)
updater.start_polling()
from main import lda_model
from telegram.ext import MessageHandler, Filters


def echo(bot, update):
    #     doc = []
    #     doc.insert(0, update.message.text)
    bot.sendMessage(chat_id=update.message.chat_id, text=lda_model(update.message.text))


echo_handler = MessageHandler(Filters.text, echo)
dispatcher.add_handler(echo_handler)
