import token_info
import telebot
import threading


class BotProxy:

    def __init__(self, bot: telebot.TeleBot) -> None:
        self._target_user = None
        self._target_chat_id = None
        self._bot = bot

    def set_target_user(self, username: str):
        self._target_user = username

    def get_target_user(self):
        return self._target_user
    
    def set_target_chat_id(self, chat_id):
        self._target_chat_id = chat_id


    def send_notification(self):
        if self._target_chat_id:
            self._bot.send_message(self._target_chat_id, "Slow down!")
        else:
            print("chat_id is not set")




def init_bot() -> BotProxy:
    bot = telebot.TeleBot(token_info.BOT_TOKEN)
    helper = BotProxy(bot)

    @bot.message_handler(commands=['help'])
    def handle_help(message):
        bot.reply_to(message, "This bot would help you control speech rate.\nTo start a session send /session command")


    @bot.message_handler(commands=['start'])
    def handle_start(message):
        bot.reply_to(message, "Welcome! This is a research project bot that helps you control your speech rate. Send /help to learn more and wait for instructions from the researchers")

    @bot.message_handler(commands=['session'])
    def handle_start_session(message):
        username = message.from_user.username
        print(username)
        if username == helper.get_target_user():
            print(f"Target user changed to {username}")
            helper.set_target_chat_id(message.chat.id)
            bot.reply_to(message, "Started")
        else:
            bot.reply_to(message, "Sorry, you session is over or hasn't started yet. Contact the researchers (@goodmove) for details.")
    
    thread = threading.Thread(None, bot.polling)
    thread.start()

    return helper


