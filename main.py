import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from textblob import TextBlob

# Task 1: Content Classifier
def content_classifier():
    data = {
        'description': [
            "A high school love story with unexpected twists.",
            "An intense rivalry in the world of competitive dance.",
            "A magical adventure where love conquers all.",
            "A humorous take on college life and friendships.",
            "A suspenseful journey through a haunted house.",
            "A heartwarming tale of family and resilience.",
            "A futuristic world where technology changes everything.",
            "A slice of life exploring deep friendships and romance.",
            "An epic showdown between good and evil forces.",
            "A historical romance set during turbulent times.",
            "A love triangle involving childhood friends.",
            "A thrilling mystery that keeps you guessing.",
            "An exploration of dreams and ambitions in a sports setting.",
            "A gripping political drama with unexpected alliances.",
            "A family-friendly time travel adventure.",
            "A survival story in a post-apocalyptic landscape."
        ],
        'category': [
            'romance', 'drama', 'fantasy', 'comedy', 'horror',
            'drama', 'sci-fi', 'slice of life', 'action', 'historical',
            'romance', 'mystery', 'sports', 'thriller', 'adventure', 'sci-fi'
        ]
    }

    df = pd.DataFrame(data)

    X = df['description']
    y = df['category']

    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)

    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    print("Task 1: Content Classifier Results")
    print(classification_report(y_test, y_pred))

# Task 2: Sentiment Analysis
def sentiment_analysis():
    comments = [
        "I love the art style!",
        "The story is boring.",
        "Amazing characters and plot!",
        "Not my cup of tea.",
        "One of the best webtoons out there!"
    ]

    positive_count = 0
    negative_count = 0

    for comment in comments:
        analysis = TextBlob(comment)
        if analysis.sentiment.polarity > 0:
            positive_count += 1
        elif analysis.sentiment.polarity < 0:
            negative_count += 1

    total_comments = len(comments)
    print("Task 2: Sentiment Analysis Results")
    print(f"Total Comments: {total_comments}")
    print(f"Positive Comments: {positive_count} ({(positive_count/total_comments) * 100:.2f}%)")
    print(f"Negative Comments: {negative_count} ({(negative_count/total_comments) * 100:.2f}%)\n")

# Task 3: Expanded Chatbot
def chatbot():
    def chatbot_response(user_input):
        responses = {
            "what is castle swimmer about": "Castle Swimmer is about a young swimmer who embarks on a journey to uncover the truth about their abilities.",
            "who are the main characters": "The main characters include the young swimmer and their mentor, who guides them through their challenges.",
            "tell me more": "The story explores themes of courage, friendship, and the pursuit of dreams.",
            "what genre is castle swimmer": "Castle Swimmer falls under the fantasy and adventure genres.",
            "how many chapters does it have": "As of now, Castle Swimmer has over 80 chapters released.",
            "what makes castle swimmer unique": "The blend of humor, drama, and fantasy elements makes it unique.",
            "who wrote castle swimmer": "Castle Swimmer is created by a talented author, whose name is not specified here.",
            "what is the main theme of castle swimmer": "The main themes are self-discovery and overcoming challenges.",
            "are there any adaptations": "As of now, there are no known adaptations for Castle Swimmer."
        }
        
        user_input = user_input.lower()
        return responses.get(user_input, "I'm sorry, I did not understand your question.")

    print("CASTLE SWIMMER CHATBOT")
    print("Type 'exit' to end the chat.")
    while True:
        user_input = input("WHAT INFORMATION ARE YOU LOOKING FOR? ")
        if user_input.lower() == 'exit':
            print("Chatbot:goodbye!im here whenever you need assistance.take care!!")
            break
        response = chatbot_response(user_input)
        print(f"Chatbot: {response}")


content_classifier()
sentiment_analysis()
chatbot()