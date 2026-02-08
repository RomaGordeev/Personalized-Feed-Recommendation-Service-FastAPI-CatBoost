from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine
import os
from pydantic import BaseModel
from typing import List
from datetime import datetime
from catboost import CatBoostClassifier
import pandas as pd


class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True

def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models():
    model_path = get_model_path("catboost_model.cbm")  
    model = CatBoostClassifier() 
    model.load_model(model_path) 
    return model

app = FastAPI()

db = (
    "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
    "postgres.lab.karpov.courses:6432/startml"
)

def get_engine():
    return create_engine(db)



def batch_load_sql(query: str) -> pd.DataFrame:
    """
    На этом этапе загружаем данные из Postgres по чанкам, чтобы не тратить слишком много
    оперативной памяти за раз
    """


    CHUNKSIZE = 200000

    engine = get_engine()
    conn = engine.connect().execution_options(stream_results=True)

    chunks = []

    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    
    conn.close()

    if not chunks:
        return pd.DataFrame()
    

    return pd.concat(chunks, ignore_index=True)

def load_user_features():
    """
    На этом этапе читаем таблицу с колонокой user_id из БД с фичами пользователей
    """

    TABLE_NAME = "arsenij_shmelev_gwh6987_features_lesson_22"  
    query = f"SELECT * FROM {TABLE_NAME}"

    user_features = batch_load_sql(query)

    user_features = user_features.set_index("user_id")


    return user_features



def load_post_features():
    """
    На этом этапе читаем таблицу с колонокой user_id из БД с фичами пользователей
    """

    TABLE_NAME = "arsenij_shmelev_gwh6987_features_lesson_22"

    engine = get_engine()

    post_text_df = pd.read_sql(
        "SELECT * FROM public.post_text_df",
        con=engine
    )

    posts_df = post_text_df[["post_id", "text", "topic"]].copy()
    posts_df = posts_df.rename(columns={"post_id": "id"})
    posts_df = posts_df.set_index("id")

    ohe_topic = pd.get_dummies(
        post_text_df["topic"],
        dummy_na=False,
        dtype=int
    )

    post_features = pd.concat(
        [post_text_df[["post_id"]], ohe_topic],
        axis=1
    )

    text_series = post_text_df["text"].fillna("")
    post_features["text_len"] = text_series.str.split().apply(len)
    post_features["unique_word_count"] = text_series.apply(
        lambda x: len(set(x.split()))
    )

    post_features["text_diversity"] = post_features.apply(
        lambda x: x["unique_word_count"] / x["text_len"] if x["text_len"] > 0 else 0,
        axis=1
    )

    N = 1000000

    feed_query = f"""
    SELECT "timestamp", user_id, post_id, target
    FROM public.feed_data
    WHERE action = 'view'
    ORDER BY "timestamp" DESC
    LIMIT {N}
    """

    feed_data = pd.read_sql(feed_query, con=engine, parse_dates=["timestamp"])

    post_stats = (
        feed_data.groupby("post_id")
        .agg(post_views=("target", "count"), post_likes=("target", "sum"))
        .reset_index()
    )

    post_stats["post_ctr"] = (
        post_stats["post_likes"] / post_stats["post_views"]
    )

    post_stats = post_stats.fillna(0)


    post_features = post_features.merge(
        post_stats,
        on="post_id",
        how="left"
    )

    for col in ["post_views", "post_likes", "post_ctr"]:
        post_features[col] = post_features[col].fillna(0)

    post_features = post_features.drop(columns=["unique_word_count"])

    post_features = post_features.set_index("post_id")

    return post_features, posts_df



model = load_models()                     
user_features = load_user_features()      
post_features, posts_df = load_post_features()

user_cols = list(user_features.columns)
post_cols = list(post_features.columns)
time_cols = ["hour", "weekday", "is_weekend"]

FEATURE_COLUMNS = time_cols + user_cols + post_cols


def make_features_user(user_id: int, time: datetime) -> pd.DataFrame:
    """
    Собираем датафрейм для отработки модели
    """

    if user_id not in user_features.index:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_row = user_features.loc[user_id]

    features = post_features.copy()

    for col in user_features.columns:
        features[col] = user_row[col]

    features["hour"] = time.hour
    features["weekday"] = time.weekday()
    features["is_weekend"] = int(features["weekday"].iloc[0] >= 5)

    X = features[FEATURE_COLUMNS].fillna(0)

    X.index.name = "post_id"

    return X


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommendation_posts(id: int, time: datetime, limit: int = 5) -> List[PostGet]:
    """
    Эндпоинт для получения топ limit рекомендаций постов для пользователя с id
    """

    X = make_features_user(id, time)

    preds = model.predict_proba(X)[:, 1]

    df_pred = pd.DataFrame(
        {
            "post_id": X.index.values,
            "pred": preds
        }
    )

    top_posts = df_pred.sort_values("pred", ascending=False).head(limit)

    recommendations: List[PostGet] = []

    for post_id in top_posts["post_id"].values:
        row = posts_df.loc[post_id]
        recommendations.append(
            PostGet(
                id=post_id,
                text=row["text"],
                topic=row["topic"]
            )
        )

    return recommendations
