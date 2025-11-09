import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
import requests as rq
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode
from bs4 import BeautifulSoup as bs
from natasha import Doc, Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger
from nltk.corpus import stopwords


class RBC_parser:
    def __init__(self,
                 query='РБК',
                 project='rbcnews',
                 category='TopRbcRu_economics',
                 material='',
                 dateFrom='2025-01-01',
                 dateTo='2025-01-01',
                 page=0):

        self.query = query
        self.project = project
        self.category = category
        self.material = material
        self.dateFrom = dateFrom
        self.dateTo = dateTo
        self.page = page
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.stop_words = stopwords.words('russian')

        self.param_dict = {
            'query': query,
            'project': project,
            'category': category,
            'dateFrom': datetime.strptime(dateFrom, '%Y-%m-%d').strftime('%d.%m.%Y'),
            'dateTo': datetime.strptime(dateTo, '%Y-%m-%d').strftime('%d.%m.%Y'),
            'page': str(page),
            'material': material
        }

    def text_prep(self, text) -> str:
        doc = Doc(text)  # get text from each row of text column
        doc.segment(self.segmenter)  # split text into sentences and tokens where each sentence is a list of tokens
        doc.tag_morph(self.morph_tagger)  # get part of word in morphological issues

        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)  # get initial word from its form

        lemmas = [_.lemma for _ in doc.tokens]  # from our lemmatize words make list
        words = [lemma for lemma in lemmas if
                 lemma.isalpha() and len(lemma) > 2]  # keep words only with only letters and length > 2
        filtered_words = [word for word in words if
                          word not in self.stop_words]  # filter specific words like articles, prepositions etc
        return " ".join(filtered_words)  # gather all together

    def _get_url(self, parameters: dict) -> str:
        base_url = "https://www.rbc.ru/search/ajax/"
        return base_url + "?" + urlencode(parameters, encoding="utf-8")

    @staticmethod
    def fetch_article(session, url, ar_date, ar_category):
        r_page = session.get(url)
        soup = bs(r_page.text, "html.parser")

        container_text = soup.find('div', class_='article__text article__text_free')
        ar_text = " ".join(
            p.get_text(strip=True) for p in container_text.find_all('p')
        ) if container_text else ""

        container_tags = soup.find('div', class_='article__tags__container')
        ar_tags = ". ".join(
            tag.get_text(strip=True) for tag in container_tags.find_all('a', class_='article__tags__item')
        ) if container_tags else ""

        return {
            'date': ar_date,
            'tags': ar_tags,
            'category': ar_category,
            'text': ar_text
        }

    def _get_data(self, max_articles=2000):
        res = []
        session = rq.Session()
        page = 0

        with ThreadPoolExecutor(max_workers=10) as executor:
            while len(res) < max_articles:
                d = {**self.param_dict, "page": str(page)}
                try:
                    response = session.get(self._get_url(d))
                    response.raise_for_status()
                    items = response.json().get('items', [])
                except Exception as e:
                    print(f"[Ошибка на странице {page}]: {e}")
                    break

                if not items:
                    break

                futures = [
                    executor.submit(
                        self.fetch_article,
                        session,
                        item.get('fronturl'),
                        item.get('publish_date'),
                        item.get('category')
                    )
                    for item in items
                ]

                for f in futures:
                    res.append(f.result())

                page += 1
                if len(res) >= max_articles:
                    break

        return pd.DataFrame(res[:max_articles])



    def get_range_data(self, save_csv=False, csv_name="default_name.csv", max_articles_per_month=2000):

        start = datetime.strptime(self.dateFrom, "%Y-%m-%d")
        end = datetime.strptime(self.dateTo, "%Y-%m-%d")

        all_dfs = []

        while start <= end:
            month_start = start.replace(day=1)
            month_end = (month_start + relativedelta(months=1)) - relativedelta(days=1)

            if month_end > end:
                month_end = end

            print(f"Статьи за {month_start.strftime('%Y-%m')}")

            parser = RBC_parser(
                query=self.query,
                project=self.project,
                category=self.category,
                material=self.material,
                dateFrom=month_start.strftime("%Y-%m-%d"),
                dateTo=month_end.strftime("%Y-%m-%d")
            )

            df_month = parser._get_data(max_articles=max_articles_per_month)
            if not df_month.empty:
                all_dfs.append(df_month)

            start = month_start + relativedelta(months=1)

        final_df = pd.concat(all_dfs, ignore_index=True)

        if save_csv:
            final_df.to_csv(csv_name, index=False, encoding="utf-8-sig")
        final_df['text_tokens'] = final_df.text.apply(self.text_prep)
        return final_df

class MeanEmbeddingVectorizer(object):
        
        def __init__(self, model):
            self.word2vec = model.wv
            self.dim = model.vector_size

        def fit(self, X, y):
            return self

        def transform(self, X):
            return np.array([
                 np.mean(
                      [self.word2vec.get_vector(w) 
                       for w in words 
                        if w in self.word2vec] or 
                            [np.zeros(self.dim)], axis=0) 
                       for words in X])