import requests
import pandas as pd
import time
import random

def fetch_books():
    queries = ["programming", "fiction"]
    books = []
    book_id_counter = 50000

    for q in queries:
        print(f"Fetching books for query: {q}")
        url = f"https://openlibrary.org/search.json?q={q}&limit=20"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            docs = data.get("docs", [])
            for doc in docs:
                title = doc.get("title", "")
                authors = "/".join(doc.get("author_name", ["Unknown"]))
                pages = doc.get("number_of_pages_median", random.randint(200, 500))
                pub_year = doc.get("first_publish_year", 2000)
                publishers = doc.get("publisher", ["Unknown"])
                publisher = publishers[0] if publishers else "Unknown"
                
                isbns = doc.get("isbn", [])
                isbn10 = isbns[0] if isbns else ""
                isbn13 = isbns[1] if len(isbns) > 1 else ""
                
                ratings_count = random.randint(100, 10000)
                
                books.append({
                    "bookID": book_id_counter,
                    "title": title,
                    "authors": authors,
                    "average_rating": 0.0, 
                    "isbn": isbn10,
                    "isbn13": isbn13,
                    "language_code": "eng",
                    "num_pages": pages,
                    "ratings_count": ratings_count,
                    "text_reviews_count": int(ratings_count * 0.1),
                    "publication_date": f"1/1/{pub_year}",
                    "publisher": publisher
                })
                book_id_counter += 1
                
        except Exception as e:
            print(f"Error fetching {q}: {e}")
            
        time.sleep(1) 

    df = pd.DataFrame(books)
    df.to_csv("test_data_for_app.csv", index=False)
    print(f"\nSuccessfully fetched {len(books)} books.")
    print("Saved to test_data_for_app.csv. You can use these rows to test the prediction app!")

if __name__ == "__main__":
    fetch_books()
