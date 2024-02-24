# %% [markdown]
# # Imports

# %%
import numpy as np
import pandas as pd
import re

# %% [markdown]
# ---

# %% [markdown]
# ### Turnbackhoax

# %%
df_turnbackhoax_1 = pd.read_csv("./turnbackhoax-1.csv", index_col=0)
df_turnbackhoax_2 = pd.read_csv("./turnbackhoax-2.csv", index_col=0)

# %%
# merge turnbackhoax csv files
df_turnbackhoax = pd.concat([df_turnbackhoax_2, df_turnbackhoax_1], ignore_index=True)

# %%
df_turnbackhoax.head()

# %% [markdown]
# Ubah format tanggal

# %%
import dateparser

df_turnbackhoax["tanggal"] = df_turnbackhoax["tanggal"].apply(lambda date_str : dateparser.parse(date_str, languages=['id']).strftime("%Y-%m-%d"))

# %% [markdown]
# Hanya ambil berita dari tanggal 1 Januari 2022 hingga 31 Agustus 2023

# %%
df_turnbackhoax = df_turnbackhoax[(df_turnbackhoax["tanggal"]<="2023-08-31") & (df_turnbackhoax["tanggal"]>="2022-01-01")]

# %%
pd.set_option('display.max_colwidth', 50)

# %%
df_turnbackhoax.head()

# %% [markdown]
# Dataframe turnbackhoax informations

# %%
df_turnbackhoax.info()

# %%
df_turnbackhoax.describe()

# %%
df_turnbackhoax.shape

# %% [markdown]
# Berapa jumlah berita hoaks yang bersumber dari media-media populer, seperti twitter, facebook, dll

# %%
list_media = ["twitter", "facebook", "whatsapp", "youtube", "tiktok", "instagram"]
turnbackhoax_count = df_turnbackhoax.count()["sumber"]
turnbackhoax_from_list_media_count = df_turnbackhoax["sumber"].str.contains("|".join(list_media), case=False, na=False).sum()
print(f"{(turnbackhoax_from_list_media_count/turnbackhoax_count)*100}%")

# %% [markdown]
# Download merged turnbackhoax

# %%
df_turnbackhoax.to_csv(r"./final_csv_data/TURNBACKHOAX-FINAL.csv",encoding="utf-8-sig")

# %% [markdown]
# ---

# %% [markdown]
# ### Detik

# %%
df_detik_1 = pd.read_csv("./detikcom-1.csv", index_col=0)
df_detik_2 = pd.read_csv("./detikcom-2.csv", index_col=0)
df_detik_3 = pd.read_csv("./detikcom-3.csv", index_col=0)
df_detik_4 = pd.read_csv("./detikcom-4.csv", index_col=0)
df_detik_5 = pd.read_csv("./detikcom-5.csv", index_col=0)

# %%
# merge detik csv files
df_detik = pd.concat([df_detik_1, df_detik_2, df_detik_3, df_detik_4, df_detik_5], ignore_index=True)

# %%
df_detik.head()

# %%
df_detik[df_detik["tanggal"].str.contains("views", case=False, na=False)]

# %% [markdown]
# Hapus "Views" dari tanggal

# %%
def detik_find_and_replace_tanggal(tgl) :
  if isinstance(tgl, str) :
    index = tgl.find("|")
    return tgl[index+3 : ] if index!=-1 else tgl
  return tgl

# %%
df_detik["tanggal"] = df_detik["tanggal"].apply(lambda tgl : detik_find_and_replace_tanggal(tgl))

# %%
df_detik.loc[14]["tanggal"]

# %% [markdown]
# Ubah format tanggal

# %%
import dateparser

df_detik["tanggal"] = df_detik["tanggal"].apply(lambda date_str : dateparser.parse(date_str, languages=["id"]).strftime("%Y-%m-%d") if pd.notna(date_str) else date_str)

# %%
df_detik.head()

# %% [markdown]
# Dataframe detik informations

# %%
df_detik.info()

# %%
df_detik.describe()

# %%
df_detik.shape

# %% [markdown]
# Download merged detik

# %%
df_detik.to_csv(r"./final_csv_data/DETIK-FINAL.csv",encoding="utf-8-sig")

# %% [markdown]
# ---

# %% [markdown]
# ### Kompas

# %%
df_kompas_1 = pd.read_csv("./kompas-1.csv", index_col=0)
df_kompas_2 = pd.read_csv("./kompas-2.csv", index_col=0)

# %%
# merge kompas csv files
df_kompas = pd.concat([df_kompas_1, df_kompas_2], ignore_index=True)

# %%
df_kompas.head()

# %% [markdown]
# Ubah format tanggal

# %%
import dateparser

df_kompas["tanggal"] = df_kompas["tanggal"].apply(lambda date_str : dateparser.parse(date_str[2:], languages=["id"]).strftime("%Y-%m-%d") if pd.notna(date_str) else date_str)

# %%
df_kompas.head()

# %% [markdown]
# Dateframe kompas informations

# %%
df_kompas.info()

# %%
df_kompas.describe()

# %%
df_kompas.shape

# %% [markdown]
# Download merged kompas

# %%
df_kompas.to_csv(r"./final_csv_data/KOMPAS-FINAL.csv",encoding="utf-8-sig")

# %% [markdown]
# ---

# %% [markdown]
# ## 1. Clean turnbackhoax

# %% [markdown]
# ##### Fungsi untuk mencari narasi pada cell narasi

# %%
def search_and_replace_pattern(dataframe, pattern) :
  for idx in dataframe.index:
    # re search pattern
    search_pattern = re.search(pattern, dataframe["narasi"][idx], re.I)
    # kalau ada pattern
    if search_pattern :
      df_turnbackhoax["narasi"][idx] = dataframe["narasi"][idx][search_pattern.end():]
    else :
      df_turnbackhoax["narasi"][idx] = ""

# %% [markdown]
# #### Hapus kolom yg tdk dibutuhkan

# %%
df_turnbackhoax.drop(["kategori", "penulis"], axis=1, inplace=True)

# %%
df_turnbackhoax.head()

# %% [markdown]
# #### Hapus baris dengan cell kosong

# %%
df_turnbackhoax.isnull().sum()

# %%
pd.set_option('display.max_colwidth', 50)

# %%
df_turnbackhoax[df_turnbackhoax["narasi"].isnull()].head()

# %%
df_turnbackhoax.dropna(subset=["narasi"], inplace=True)

print(df_turnbackhoax.isnull().sum())

# %% [markdown]
# ### Hapus data double

# %%
df_turnbackhoax.drop_duplicates(subset=["judul"], inplace=True)
df_turnbackhoax.drop_duplicates(subset=["narasi"], inplace=True)

# %%
df_turnbackhoax.count()

# %% [markdown]
# Kolom "tanggal" tidak diperlukan pada dataset turnbackhoax karena data akan diambil semua

# %%
df_turnbackhoax.drop(["tanggal", "sumber"], axis=1, inplace=True)

# %%
display(df_turnbackhoax)

# %% [markdown]
# ### Pembersihan data2 yg tidak dibutuhkan

# %% [markdown]
# #### Handle narasi yg dimulai dari kata2 berikut

# %%
narasi_pattern = r"=*\[?narasi\]?\s*[:]?"

# %%
narasi_wrong_startswith = df_turnbackhoax.loc[df_turnbackhoax["narasi"].str.startswith(("Tidak","tidak", "Informasi", "informasi", "Video", "Hasil periksa", "hasil periksa", "Beredar", "Konten", "Unggahan", "dalam", "Dalam", "Judul", "\nJudul", "judul", "JUDUL", "Situs palsu"))]

display(narasi_wrong_startswith)

# %%
search_and_replace_pattern(narasi_wrong_startswith, narasi_pattern)

# %% [markdown]
# #### Handle narasi yang mengandung kata2 berikut

# %%
narasi_wrong_contains = df_turnbackhoax.loc[df_turnbackhoax["narasi"].str.contains("akun|narasi dalam foto|faktanya|kategori", case=False)] 
display(narasi_wrong_contains)

# %%
search_and_replace_pattern(narasi_wrong_contains, narasi_pattern)

# %% [markdown]
# #### Handle berita terjemahan

# %%
diterjemahkan_pattern = r"\(diterjemahkan ke bahasa indonesia\):?"

# %%
diterjemahkan = df_turnbackhoax.loc[df_turnbackhoax["narasi"].str.contains("(diterjemahkan)", case=False)]
display(diterjemahkan)

# %%
search_and_replace_pattern(diterjemahkan, diterjemahkan_pattern)

# %%
terjemahan_pattern = r"terjemahan:?"

# %%
terjemahan = df_turnbackhoax.loc[df_turnbackhoax["narasi"].str.contains("terjemahan", case=False)] 
display(terjemahan)

# %%
search_and_replace_pattern(terjemahan, terjemahan_pattern)

# %% [markdown]
# #### Handle berita yang mengandung kata "klaim"

# %%
klaim_pattern = r"\bKlaim\b"
klaim_salah = df_turnbackhoax.loc[df_turnbackhoax["narasi"].str.contains(klaim_pattern, case=False)]
search_and_replace_pattern(klaim_salah, narasi_pattern)

# %% [markdown]
# #### Handle berita yang mengandung kata "sumber"

# %%
sumber_salah = df_turnbackhoax.loc[df_turnbackhoax["narasi"].str.contains(r"=*\[?sumber\]?\s*[:]?", case=False)]

# %%
for idx in sumber_salah.index:
  search_pattern = re.search("sumber", sumber_salah["narasi"][idx], re.I)
  if search_pattern :
    df_turnbackhoax["narasi"][idx] = sumber_salah["narasi"][idx][:search_pattern.start()]
  else :
    df_turnbackhoax["narasi"][idx] = ""

# %%
bagian_referensi_pattern = "[\(\[]narasi\s?(dilanjutkan|lanjutan|selanjutnya)?\s?(ada di|di)?\s?bagian referensi[\)\]]"
bagian_referensi_salah = df_turnbackhoax.loc[df_turnbackhoax["narasi"].str.contains(bagian_referensi_pattern, case=False)]

for idx in bagian_referensi_salah.index:
  search_pattern = re.search(bagian_referensi_pattern, bagian_referensi_salah["narasi"][idx], re.I)
  if search_pattern :
    df_turnbackhoax["narasi"][idx] = bagian_referensi_salah["narasi"][idx][:search_pattern.start()]
  else :
    df_turnbackhoax["narasi"][idx] = ""


# %%
df_turnbackhoax.eq("").sum()

# %%
# df_turnbackhoax!="" artinya apabila cell tersebut bukan empty string, maka akan true.

df_turnbackhoax = df_turnbackhoax[df_turnbackhoax!=""].dropna()

# %%
df_turnbackhoax[df_turnbackhoax["narasi"].str.len() <= 1]

# %%
df_turnbackhoax = df_turnbackhoax[df_turnbackhoax["narasi"].str.len() > 1]

# %%
df_turnbackhoax.count()

# %% [markdown]
# ### Download cleaned Turnbackhoax data

# %%
df_turnbackhoax.to_csv(r"./new_csv_files/Cleaned-Turnbackhoax.csv", encoding="utf-8-sig")

# %% [markdown]
# ---

# %% [markdown]
# ## 2. Clean detik

# %%
df_detik.drop(["kategori", "penulis"], axis=1, inplace=True)
display(df_detik)

# %%
df_detik.isnull().sum()

# %%
display(df_detik[df_detik["narasi"].isnull()])

# %%
df_detik.dropna(subset=["narasi"], inplace=True)

print(df_detik.isnull().sum())

# %%
df_detik.drop_duplicates(subset=["judul"], inplace=True)
df_detik.drop_duplicates(subset=["narasi"], inplace=True)

# %%
df_detik.count()

# %%
df_detik.loc[df_detik["narasi"].str.contains("[Gambas:Video 20detik]", regex=False, case=False)]

# %%
gambas = df_detik.loc[df_detik["narasi"].str.contains("[Gambas:Video 20detik]", regex=False, case=False)]

for idx in gambas.index:
  search_pattern = re.search("\[Gambas:Video 20detik\]", gambas["narasi"][idx], re.I)
  if search_pattern :
    df_detik["narasi"][idx] = gambas["narasi"][idx][:search_pattern.start()]

display(df_detik)

# %% [markdown]
# ### Download cleaned Detik data

# %%
df_detik.to_csv(r"./new_csv_files/Cleaned-Detik.csv", encoding="utf-8-sig")

# %% [markdown]
# ---

# %% [markdown]
# ## 3. Clean Kompas

# %%
df_kompas.isnull().sum()

# %%
display(df_kompas[df_kompas["narasi"].isnull()])

# %%
df_detik.dropna(subset=["narasi"], inplace=True)

print(df_detik.isnull().sum())

# %%
df_kompas.drop(["kategori", "penulis"], axis=1, inplace=True)
display(df_kompas)

# %% [markdown]
# ### Download cleaned Kompas data

# %%
df_kompas.to_csv(r"./new_csv_files/Cleaned-Kompas.csv",encoding="utf-8-sig")

# %% [markdown]
# ---

# %% [markdown]
# ### Data selection (pilih data yang akan diambil untuk df_hoaks dan df_faktual)

# %%
from urllib.parse import quote

def create_date_list(start_date, end_date) :
  return [quote(date.strftime("%Y-%m-%d"), safe="") for date in pd.date_range(start=start_date, end=end_date)]

# %%
# ambil semua berita hoaks dari turnbackhoax
df_hoaks = df_turnbackhoax

df_faktual = pd.DataFrame(columns=["judul", "narasi", "tanggal", "sumber"])

list_date = create_date_list("2022-01-01","2023-08-31")

# total berita yang harus diambil per hari = 5
for date in list_date :
  # seluruh berita detik dan kompas pada tanggal tersebut
  detik = df_detik.loc[df_detik["tanggal"]==date]
  kompas = df_kompas.loc[df_kompas["tanggal"]==date]
  
  day = int(date.split('-')[-1])

  if day%2 == 0 :
    # jika hari genap, ambil tiga berita dari detik, dan dua berita dari kompas
    df_faktual = pd.concat([df_faktual, df_detik.loc[df_detik["tanggal"]==date].iloc[:3]], ignore_index=True)
    df_faktual = pd.concat([df_faktual, df_kompas.loc[df_kompas["tanggal"]==date].iloc[:2]], ignore_index=True)
  else :
    # jika hari ganjil, ambil dua berita dari detik, dan tiga berita dari kompas
    df_faktual = pd.concat([df_faktual, df_detik.loc[df_detik["tanggal"]==date].iloc[:2]], ignore_index=True)
    df_faktual = pd.concat([df_faktual, df_kompas.loc[df_kompas["tanggal"]==date].iloc[:3]], ignore_index=True)

# %%
df_faktual.head(10)

# %% [markdown]
# ### Download df_hoaks dan df_faktual

# %%
df_faktual.drop(["tanggal", "sumber"], axis=1, inplace=True)

# %%
df_hoaks.to_csv(r"./new_csv_files/HOAKS.csv",encoding="utf-8-sig")
df_faktual.to_csv(r"./new_csv_files/FAKTUAL.csv",encoding="utf-8-sig")

# %% [markdown]
# ---

# %% [markdown]
# ## 4. Gabung kolom "judul" dan "narasi" menjadi "judul_dan_narasi", dan tambahkan label

# %%
pd.set_option('display.max_colwidth', 120)

# %%
df_hoaks = pd.read_csv("./new_csv_files/HOAKS.csv", index_col=0)
df_faktual = pd.read_csv("./new_csv_files/FAKTUAL.csv", index_col=0)

# %%
df_hoaks.loc[:,"judul_dan_narasi"] = df_hoaks["judul"] + " " + df_hoaks["narasi"]
df_faktual.loc[:,"judul_dan_narasi"] = df_faktual["judul"] + " " + df_faktual["narasi"]

# %%
display(df_hoaks)

# %% [markdown]
# ### Drop kolom "judul" dan "narasi"

# %%
df_hoaks.drop(["judul", "narasi"], axis=1, inplace=True)
df_faktual.drop(["judul", "narasi"], axis=1, inplace=True)

# %%
display(df_hoaks)

# %%
pd.set_option('display.max_colwidth', 100)

# %%
display(df_faktual)

# %% [markdown]
# ### Tambahkan label 1 pada turnbackhoax, dan label 0 pada detik dan liputan

# %%
df_hoaks.insert(len(df_hoaks.columns), "label", 1, False)
df_faktual.insert(len(df_faktual.columns), "label", 0, False)

# %%
display(df_hoaks)
display(df_faktual)

# %% [markdown]
# ### Gabung kedua dataframe tersebut

# %%
df = pd.concat([df_hoaks, df_faktual], ignore_index=True)

# %%
display(df)

# %% [markdown]
# ### Download hasil cleansing

# %%
df.to_csv(r"./new_csv_files/GABUNGAN.csv", encoding="utf-8-sig")

# %% [markdown]
# ---

# %% [markdown]
# # Step 2 : Text Preprocessing

# %%
df = pd.read_csv("./new_csv_files/GABUNGAN.csv", index_col=0)

# %%
display(df)

# %%
pd.set_option('display.max_colwidth', 120)

# %%
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

# %% [markdown]
# ### Function utk mapping special uppercase ke standard lowercase

# %%
def special_to_standard_lowercase(s):
  # bold upper dan lower
  bold_upper_lower = "𝗔𝗕𝗖𝗗𝗘𝗙𝗚𝗛𝗜𝗝𝗞𝗟𝗠𝗡𝗢𝗣𝗤𝗥𝗦𝗧𝗨𝗩𝗪𝗫𝗬𝗭𝗮𝗯𝗰𝗱𝗲𝗳𝗴𝗵𝗶𝗷𝗸𝗹𝗺𝗻𝗼𝗽𝗾𝗿𝘀𝘁𝘂𝘃𝘄𝘅𝘆𝘇"
  # monospace upper dan lower
  monospace_upper_lower = "𝙰𝙱𝙲𝙳𝙴𝙵𝙶𝙷𝙸𝙹𝙺𝙻𝙼𝙽𝙾𝙿𝚀𝚁𝚂𝚃𝚄𝚅𝚆𝚇𝚈𝚉𝚊𝚋𝚌𝚍𝚎𝚏𝚐𝚑𝚒𝚓𝚔𝚕𝚖𝚗𝚘𝚙𝚚𝚛𝚜𝚝𝚞𝚟𝚠𝚡𝚢𝚣"
  
  regular_chars = "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"

  translation = str.maketrans(bold_upper_lower + monospace_upper_lower, regular_chars*2)

  # translate string dan lower hasilnya
  return s.translate(translation).lower()

# %% [markdown]
# ### Function preprocessing yang dipisah (untuk dokumen skripsi)

# %%
display(df)

# %%
# df['judul_dan_narasi'] = df['judul_dan_narasi'].apply(lambda x : special_to_standard_lowercase(x))

# from nltk.tokenize import wordpunct_tokenize
# df['judul_dan_narasi'] = df['judul_dan_narasi'].apply(lambda x : wordpunct_tokenize(x))

# from nltk.corpus import stopwords
# STOPWORDS = set(stopwords.words("indonesian") + stopwords.words("english"))
# df['judul_dan_narasi'] = df['judul_dan_narasi'].apply(lambda x : list(filter(lambda t : t.lower() not in STOPWORDS, x)))


# import string
# PUNCTUATIONS = set(string.punctuation)
# PUNCTUATIONS.add('“')
# PUNCTUATIONS.add('”')
# df['judul_dan_narasi'] = df['judul_dan_narasi'].apply(lambda x : list(filter(lambda t : not all (char in PUNCTUATIONS for char in t), x)))


# df['judul_dan_narasi'] = df['judul_dan_narasi'].apply(lambda x : list(filter(lambda t : not (t.isnumeric() or t[0].isnumeric()), x)))


# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# factory = StemmerFactory()
# stemmer = factory.create_stemmer()
# df['judul_dan_narasi'] = df['judul_dan_narasi'].apply(lambda x : stemmer.stem(" ".join(x)))


# %%
display(df)

# %% [markdown]
# ### Function untuk melakukan text preprocessing

# %%
def detect_and_remove_numbers(text):
  text = re.sub(r"^\d+", "", text)  # hapus leading numbers
  text = re.sub(r"\d+$", "", text)  # hapus trailing numbers
  return text

# %%
def text_preprocessing(text) :
  # case folding
  text = special_to_standard_lowercase(text)

  # remove urls
  text = re.sub(r"https?://[^\s]+", r"", text)

  # tokenizing
  text = wordpunct_tokenize(text)

  # filtering
  # stoplist/stopwords
  STOPWORDS = set(stopwords.words("indonesian") + stopwords.words("english"))
  text = list(filter(lambda t : t.lower() not in STOPWORDS, text))

  # punctuational
  PUNCTUATIONS = set(string.punctuation)
  PUNCTUATIONS.add('“')
  PUNCTUATIONS.add('”')
  text = list(filter(lambda t : not all (char in PUNCTUATIONS for char in t), text))

  # stemming
  text = list(stemmer.stem(t) for t in text)
  
  # number removal
  text = list(detect_and_remove_numbers(t) for t in text)

  return " ".join(text)

# %% [markdown]
# ### Apply function text preprocessing terhadap kolom judul_dan_narasi

# %%
df['judul_dan_narasi'] = df['judul_dan_narasi'].apply(lambda x : text_preprocessing(x))

# %%
display(df)

# %% [markdown]
# ### Download data hasil preprocessing

# %%
df.to_csv(r"./new_csv_files/data-setelah-preprocessing.csv", encoding="utf-8-sig")

# %% [markdown]
# ---

# %% [markdown]
# # Step 3 : Text Processing

# %%
import pandas as pd

# %%
pd.set_option('display.max_colwidth', 120)

# %%
df = pd.read_csv("./new_csv_files/data-setelah-preprocessing.csv", index_col=0)

# %%
display(df)

# %% [markdown]
# ## 1. Split data to training & testing data

# %%
from sklearn.model_selection import train_test_split

X = df.judul_dan_narasi
y = df.label

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# %% [markdown]
# #### Training data

# %%
display(x_train)

# %% [markdown]
# #### Testing data

# %%
display(x_test)

# %% [markdown]
# ## 2. Fit and transform training data to Tf-idf vector

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorized_x_train = vectorizer.fit_transform(x_train)

# %%
print(vectorized_x_train)

# %% [markdown]
# ### Ubah matriks menjadi dataframe

# %%
vectorized_x_train_df = pd.DataFrame(vectorized_x_train.toarray(), columns = vectorizer.get_feature_names_out())

# %%
display(vectorized_x_train_df)

# %% [markdown]
# ## 3. LOGISTIC REGRESSION model fitting ke data training

# %%
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(vectorized_x_train, y_train)

# %% [markdown]
# ### Prediksi training data dengan model tsb

# %%
lr_x_train_prediction = lr.predict_proba(vectorized_x_train)

# %%
print(lr_x_train_prediction)

# %% [markdown]
# ### Tambahkan kolom "probabilitas berita asli" (0) dan "probabilitas berita palsu" (1) pada data training

# %%
lr_x_train_prediction_df = pd.DataFrame(lr_x_train_prediction, columns = ["probabilitas berita asli", "probabilitas berita palsu"])

# %%
display(lr_x_train_prediction_df)

# %% [markdown]
# #### Apabila probabilitas tinggi, artinya kemungkinan besar berita palsu, dan sebaliknya

# %%
x_train = pd.concat([vectorized_x_train_df, lr_x_train_prediction_df["probabilitas berita palsu"]], axis=1)

# %%
display(x_train)

# %% [markdown]
# ## 4. SVM model fitting ke data training

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

# %%

def display_evaluation(model, y, y_pred) :
  cm = confusion_matrix(y, y_pred)
  cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

  print(f"Accuracy    : {accuracy_score(y, y_pred)}")
  print(f"Precision   : {precision_score(y, y_pred)}")
  print(f"Recall      : {recall_score(y, y_pred)}")
  print(f"F1-score    : {f1_score(y, y_pred)}")

  cmd.plot()

# %%
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear')
classifier.fit(x_train, y_train)

# %%
print("Coefficients:")
for feature, coef in zip(range(x_train.shape[1]), classifier.coef_[0]):
    print(f"Feature {feature+1}: {coef}")

print("\nIntercept:", classifier.intercept_[0])

# %% [markdown]
# ### Prediksi training data dengan model tsb

# %%
y_train_prediction = classifier.predict(x_train)

# %%
display_evaluation(classifier, y_train, y_train_prediction)

# %% [markdown]
# ---

# %% [markdown]
# # Step 4 : Prediksi data testing

# %% [markdown]
# ## 1. Transform testing data dengan model tf-idf tadi

# %%
vectorized_x_test = vectorizer.transform(x_test)coe

# %%
vectorized_x_test_df = pd.DataFrame(vectorized_x_test.toarray(), columns = vectorizer.get_feature_names_out())

# %%
display(vectorized_x_test_df)

# %% [markdown]
# ### Tambahkan kolom "probabilitas berita palsu" pada model logistic regression tadi

# %%
lr_x_test_prediction = lr.predict_proba(vectorized_x_test)

# %%
lr_x_test_prediction_df = pd.DataFrame(lr_x_test_prediction, columns = ["probabilitas berita asli", "probabilitas berita palsu"])

# %%
display(lr_x_test_prediction_df)

# %%
x_test = pd.concat([vectorized_x_test_df, lr_x_test_prediction_df["probabilitas berita palsu"]], axis=1)

# %%
display(x_test)

# %%
y_test_prediction = classifier.predict(x_test)

# %%
from sklearn.metrics import classification_report

print(classification_report(y_test, y_test_prediction))

# %%
display_evaluation(classifier, y_test, y_test_prediction)

# %% [markdown]
# ---

# %% [markdown]
# # Visualization

# %%
import pandas as pd
import numpy as np

# %%
df_after_preprocess = pd.read_csv("./new_csv_files/data-setelah-preprocessing.csv", index_col=0)

# %% [markdown]
# ### Line chart perbandingan akurasi, presisi, sensitivitas, dan f1-score pada rasio pembagian data yang berbeda-beda

# %%
import matplotlib.pyplot as plt

y = [[94.3, 95.3, 95.7, 96.1, 96.1, 96.5, 96.7, 96.4, 96.4], [95.6, 96.3, 96.7, 96.6, 96.9, 96.9, 97.1, 96.7, 96.5], [92.4, 94, 94.3, 95.4, 94.8, 95.7, 96, 95.7, 96.2], [94, 95.1, 95.5, 96, 95.8, 96.3, 96.5, 96.2, 96.3]]
x = ["10:90", "20:80", "30:70", "40:60", "50:50", "60:40", "70:30", "80:20", "90:10"]

plt.plot(x, y[0], label="akurasi", color="blue")
plt.plot(x, y[1], label="presisi", color="red")
plt.plot(x, y[2], label="sensitivitas", color="green")
plt.plot(x, y[3], label="f1-score", color="black")

plt.title("Line chart perbandingan metrik evaluasi pada rasio pembagian yang berbeda-beda", fontsize=10)
plt.xlabel("Rasio pembagian")
plt.ylabel("nilai (%)")
plt.ylim([85, 100])

plt.legend(loc="lower left")
plt.show()

# %% [markdown]
# ### Line chart perbandingan akurasi, presisi, sensitivitas, dan f1-score pada random_state pembagian data yang berbeda-beda

# %%
import matplotlib.pyplot as plt

y = [[96.4, 96.8, 97.1, 97, 97.5, 97.1, 97.1, 96.9, 98.1, 97.6], [96.7, 96.7, 97.1, 97.1, 97.3, 97.4, 96.8, 97.6, 97.8, 97.9], [95.7, 96.7, 97.1, 96.9, 97.3, 96.7, 97.2, 95.6, 98.2, 97.2], [96.2, 96.7, 97.1, 97, 97.3, 97.1, 97, 96.6, 98, 97.5]]
x = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

plt.plot(x, y[0], label="akurasi", color="blue")
plt.plot(x, y[1], label="presisi", color="red")
plt.plot(x, y[2], label="sensitivitas", color="green")
plt.plot(x, y[3], label="f1-score", color="black")

plt.title("Line chart perbandingan metrik evaluasi pada random_state yang berbeda-beda", fontsize=10)
plt.xlabel("random_state")
plt.ylabel("nilai (%)")
plt.ylim([90, 100])

plt.legend(loc="lower left")
plt.show()

# %% [markdown]
# ### 20 Kata terbanyak untuk hoaks vs faktual

# %%
df_hoaks = df_after_preprocess.loc[df_after_preprocess["label"] == 1]
df_faktual = df_after_preprocess.loc[df_after_preprocess["label"] == 0]

# %%
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer()
X = count_vectorizer.fit_transform(df_hoaks["judul_dan_narasi"])
sums = X.sum(axis=0).A1
sorted_hoaks_sums = np.argsort(sums)[::-1][:20]

for iter, sorted in enumerate(sorted_hoaks_sums) : 
  print(f"Word with {sums[sorted]} amount : {count_vectorizer.get_feature_names_out()[sorted]}")


# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(15,10))

labels = [count_vectorizer.get_feature_names_out()[sum] for sum in sorted_hoaks_sums]
heights = [sums[sum] for sum in sorted_hoaks_sums]

plt.title("Bar chart distribusi frekuensi kata berita hoaks", fontsize=16)
plt.xlabel("Frekuensi", fontsize=12)
plt.ylabel("Kata", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.barh(labels, heights, color="tab:red")

plt.gca().invert_yaxis()

plt.show()

# %%
count_vectorizer_2 = CountVectorizer()
X = count_vectorizer_2.fit_transform(df_faktual["judul_dan_narasi"])

sums = X.sum(axis=0).A1

sorted_faktual_sums = np.argsort(sums)[::-1][:20]

for iter, sorted in enumerate(sorted_faktual_sums) : 
  print(f"Word with {sums[sorted]} amount : {count_vectorizer_2.get_feature_names_out()[sorted]}")

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(15,10))

labels = [count_vectorizer_2.get_feature_names_out()[sum] for sum in sorted_faktual_sums]
heights = [sums[sum] for sum in sorted_faktual_sums]

plt.title("Bar chart distribusi frekuensi kata berita faktual", fontsize=16)
plt.xlabel("Kata", fontsize=12)
plt.ylabel("Frekuensi", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.barh(labels, heights, color="tab:blue")

plt.gca().invert_yaxis()
plt.show()

# %% [markdown]
# ### perbandingan kata-kata yang umum muncul pada berita hoaks pada berita hoaks dan berita faktual

# %%
# gempar
total_gempar = df_after_preprocess["judul_dan_narasi"].str.count("gempar").sum()
print(total_gempar)
total_gempar_hoaks = df_hoaks["judul_dan_narasi"].str.count("gempar").sum()
print(total_gempar_hoaks)
total_gempar_faktual = df_faktual["judul_dan_narasi"].str.count("gempar").sum()
print(total_gempar_faktual)

print("---")

# breaking news
total_breaking = df_after_preprocess["judul_dan_narasi"].str.count("breaking news").sum()
print(total_breaking)
total_breaking_hoaks = df_hoaks["judul_dan_narasi"].str.count("breaking news").sum()
print(total_breaking_hoaks)
total_breaking_faktual = df_faktual["judul_dan_narasi"].str.count("breaking news").sum()
print(total_breaking_faktual)

print("---")


# geger
total_geger = df_after_preprocess["judul_dan_narasi"].str.count("geger").sum()
print(total_geger)

total_geger_hoaks = df_hoaks["judul_dan_narasi"].str.count("geger").sum()
print(total_geger_hoaks)

total_geger_faktual = df_faktual["judul_dan_narasi"].str.count("geger").sum()
print(total_geger_faktual)

print("---")

# kejut
total_kejut = df_after_preprocess["judul_dan_narasi"].str.count("kejut").sum()
print(total_kejut)

total_kejut_hoaks = df_hoaks["judul_dan_narasi"].str.count("kejut").sum()
print(total_kejut_hoaks)

total_kejut_faktual = df_faktual["judul_dan_narasi"].str.count("kejut").sum()
print(total_kejut_faktual)

print("---")


# cekam
total_cekam = df_after_preprocess["judul_dan_narasi"].str.count("cekam").sum()
print(total_cekam)

total_cekam_hoaks = df_hoaks["judul_dan_narasi"].str.count("cekam").sum()
print(total_cekam_hoaks)

total_cekam_faktual = df_faktual["judul_dan_narasi"].str.count("cekam").sum()
print(total_cekam_faktual)

print("---")


# detik-detik
total_detik = df_after_preprocess["judul_dan_narasi"].str.count("detik-detik|detik detik").sum()
print(total_detik)

total_detik_hoaks = df_hoaks["judul_dan_narasi"].str.count("detik-detik|detik detik").sum()
print(total_detik_hoaks)

total_detik_faktual = df_faktual["judul_dan_narasi"].str.count("detik-detik|detik detik").sum()
print(total_detik_faktual)

# %%
# Sample data
labels = ["gempar", "breaking news", "geger", "kejut", "cekam", "detik-detik"]
hoax_values = [total_gempar_hoaks, total_breaking_hoaks, total_geger_hoaks, total_kejut_hoaks, total_cekam_hoaks, total_detik_hoaks]
faktual_values = [total_gempar_faktual, total_breaking_faktual, total_geger_faktual, total_kejut_faktual, total_cekam_faktual, total_detik_faktual]

y = np.arange(len(labels))
width = 0.35  

# Create the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.barh(y - width/2, hoax_values, width, label="Hoaks", color='tab:red')
rects2 = ax.barh(y + width/2, faktual_values, width, label="Faktual", color='tab:blue')

# Tambahkan title, label, dan ticks
ax.set_title("Perbandingan frekuensi kemunculan kata-kata yang sangat umum pada berita hoaks")
ax.set_xlabel('frekuensi')
ax.set_ylabel('kata')
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=12)
ax.legend()

# tambahkan anotasi
def autolabel(rects):
    for rect in rects:
        width = rect.get_width()
        ax.annotate('{}'.format(width),
                    xy=(width, rect.get_y() + rect.get_height() / 2),
                    xytext=(20, 0),  # 20 points horizontal offset
                    textcoords="offset points",
                    ha='center', va='center')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()


# %% [markdown]
# ### Perbandingan rasio penyebaran berita hoaks pada media-media

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 7))

labels = ["facebook", "twitter", "youtube", "whatsapp", "tiktok", "instagram"]
colors = ["#3b5998", "#00acee", "#c4302b", "#25d366", "black", "#d62976"]
heights = [28.9, 17.3, 17.27, 9.68, 6.05, 1.89]

plt.barh(labels, heights, color=colors)
plt.gca().invert_yaxis()
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)


for index, value in enumerate(heights):
  plt.text(value, index, ' ' + str(value) + "%")

plt.show()

# %% [markdown]
# ### Perbandingan jumlah data hoaks dan non hoaks

# %%
import matplotlib.pyplot as plt

labels = [0,1]
heights = [len(df_after_preprocess.loc[df_after_preprocess["label"] == 0]), len(df_after_preprocess.loc[df_after_preprocess["label"] == 1])]
colors = ["tab:blue", "tab:red"]

plt.title("Perbandingan jumlah data berita hoaks dengan berita faktual")
plt.xlabel("Label")
plt.ylabel("Jumlah data")
plt.bar(labels, heights, color=colors, label=["berita faktual", "berita hoaks"])
plt.legend()
plt.xticks([0,1])
plt.show()

# %%
pd.set_option('display.max_colwidth', 50)

# %%
df_after_preprocess.drop_duplicates(inplace=True)

# %%
print(df_after_preprocess.loc[df_after_preprocess["label"] == 0]["judul_dan_narasi"].describe())
print("\n")
print(df_after_preprocess.loc[df_after_preprocess["label"] == 1]["judul_dan_narasi"].describe())


# %% [markdown]
# ### Wordcloud utk berita hoaks dan faktual

# %%
from wordcloud import WordCloud

text = " ".join(txt for txt in df_after_preprocess["judul_dan_narasi"].loc[df_after_preprocess["label"] == 1])
wordcloud = WordCloud(
  width=400,
  height=200,
  max_words=200,
  background_color="white"
).generate(text)

plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# %%
text = " ".join(txt for txt in df_after_preprocess["judul_dan_narasi"].loc[df_after_preprocess["label"] == 0])
wordcloud = WordCloud(
  width=400,
  height=200,
  max_words=200,
  background_color="white"
).generate(text)

plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# %% [markdown]
# ### Perbandingan rata-rata panjang teks data hoaks dan non hoaks

# %%
df_after_preprocess["length"] = df_after_preprocess["judul_dan_narasi"].apply(lambda x: len(x) - x.count(" "))
df_after_preprocess

# %%
bins = np.linspace(0, 1500, 40)

plt.hist(df_after_preprocess[df_after_preprocess["label"]== 1]["length"], bins, alpha=0.5, label="Hoaks", color="tab:red")
plt.hist(df_after_preprocess[df_after_preprocess["label"]== 0]["length"], bins, alpha=0.5, label="Faktual", color="tab:blue")
plt.title("Histogram perbandingan panjang berita hoaks dan berita faktual")
plt.xlabel("Panjang berita")
plt.ylabel("Jumlah berita")
plt.legend()
plt.show()

# %%
hoax_len_avg = df_after_preprocess.loc[df_after_preprocess["label"]==1]["length"].mean()
real_len_avg = df_after_preprocess.loc[df_after_preprocess["label"]==0]["length"].mean()

print(f"Panjang rata-rata berita hoaks : {hoax_len_avg}")
print(f"Panjang rata-rata berita faktual : {real_len_avg}")


# %% [markdown]
# ### 20 fitur teratas untuk berita asli dan berita hoaks (logistic)

# %%
x_final = pd.read_csv("./csv_files/data-setelah-logistic_regression.csv", index_col=0)

# %%
# feature_names = np.array(vectorizer.get_feature_names_out())
# coefficients = lr.coef_[0]

feature_names = np.array(x_final.columns)
coefficients = classifier.coef_[0]

# %%
# top 20
top_n = 20

top_positive_coef = np.argsort(coefficients)[-top_n:]
top_negative_coef = np.argsort(coefficients)[:top_n]
top_coefficients = np.hstack([top_negative_coef, top_positive_coef])

# %%
print(coefficients)

# %%
print(top_negative_coef)

# %%
# feature names and its coefficients
print(f"Name  : {feature_names[top_negative_coef[0]]}, coefficient   : {coefficients[top_negative_coef[0]]}")

# %%
plt.figure(figsize=(15,7))
colors = ["blue" if coef<0 else "red" for coef in coefficients[top_coefficients]]
plt.bar(np.arange(2*top_n), coefficients[top_coefficients], color=colors)
plt.xticks(np.arange(2*top_n), feature_names[top_coefficients], rotation=60, ha='right')
plt.show()


