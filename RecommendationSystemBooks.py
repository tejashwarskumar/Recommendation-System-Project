import pandas as pd
book = pd.read_csv("C:/My Files/Excelr/09 - Recommendation Systems/Assignment/Book1.csv")
book.shape
book.columns
book.Publisher

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words="english") # Creating a Tfidf Vectorizer to remove all stop words
book["Publisher"].isnull().sum() 
book["Publisher"] = book["Publisher"].fillna(" ")

tfidf_matrix = tfidf.fit_transform(book.Publisher)
tfidf_matrix.shape

from sklearn.metrics.pairwise import linear_kernel
cosine_sim_matrix = linear_kernel(tfidf_matrix,tfidf_matrix)

book_index = pd.Series(book.index,index=book['BookTitle']).drop_duplicates()
book_index["The Mummies of Urumchi"]

def get_book_recommendations(BookTitle,topN):
    book_id = book_index[BookTitle]
    cosine_scores = list(enumerate(cosine_sim_matrix[book_id]))
    cosine_scores = sorted(cosine_scores,key=lambda x:x[1],reverse = True)
    cosine_scores_10 = cosine_scores[0:topN+1]
    book_idx  =  [i[0] for i in cosine_scores_10]
    book_scores =  [i[1] for i in cosine_scores_10]
    
    # Similar movies and scores
    book_similar_show = pd.DataFrame(columns=["BookTitle","Score"])
    book_similar_show["BookTitle"] = book.loc[book_idx,"BookTitle"]
    book_similar_show["Score"] = book_scores
    book_similar_show.reset_index(inplace=True)  
    book_similar_show.drop(["index"],axis=1,inplace=True)
    print (book_similar_show)

get_book_recommendations("The Mummies of Urumchi",topN=15)
