import logging

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from natasha import Doc, Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger
import nltk
import joblib
from nltk.corpus import stopwords
from news_parse.parser import RBC_parser, MeanEmbeddingVectorizer
import asyncio
from aiogram import Bot, Dispatcher, types
import asyncio
import logging
import io
import re
from datetime import date

import matplotlib
matplotlib.use("agg")  # headless backend for servers
import matplotlib.pyplot as plt
import seaborn as sns  # if you use it in your code

from aiogram import Bot, Dispatcher, types
from aiogram.types import BufferedInputFile


BOT_TOKEN = "8475081720:AAE_jd95QSAdWVC-QmxUw9Mw5AYMsPlBMUY"
RE_DDMMYY = re.compile(r'^(\d{4})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$')
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
pipe = joblib.load("/Users/joy/ml_sandbox/t_nlp/models/4000_doc.pkl")

@dp.message()
async def echo_message(message: types.Message):
    await message.answer("You can type a news date in format YYYY-MM-DD to get prediction results")

@dp.message()
async def new_message(message: types.Message, p=pipe):
    if RE_DDMMYY.fullmatch(message.text):
        user_date = message.text
        data = RBC_parser(dateFrom=user_date, dateTo=user_date).get_range_data(max_articles_per_month=50)
        cm = confusion_matrix(data['category'], pipe.predict(data['text_tokens']), labels=pipe.classes_)
        cm_df = pd.DataFrame(data=cm, columns=pipe.classes_, index=pipe.classes_)
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.heatmap(cm_df, square=False, annot=True, cmap='Blues', fmt='d', cbar=False)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
        plt.close(fig)
        buf.seek(0)
        image = BufferedInputFile(buf.getvalue(), filename="confusion_matrix.png")
        await message.answer_photo(photo=image, caption="Here’s the results")
    else:
        await message.answer("wrong format must be day month year")

async def main():
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)



if __name__ == "__main__":
    asyncio.run(main())