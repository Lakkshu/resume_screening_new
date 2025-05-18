import streamlit as st
import pickle
# import docx  # Extract text from Word file
import PyPDF2  # Extract text from PDF
import re
import nltk  # Natural Language Toolkit
# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf = TfidfVectorizer()

nltk.download('punkt')
nltk.download('stopwords')

# loading models
# clf = pickle.load(open('clf.pkl', 'rb'))
# tfidfd = pickle.load(open('tfidf.pkl', 'rb'))
try:
    clf = pickle.load(open('clf.pkl', 'rb'))
    tfidfd = pickle.load(open('tfidf.pkl', 'rb'))
    def cleanResume(txt):
        cleanText = re.sub('http\S+\s', ' ', txt)
        cleanText = re.sub('RT|cc', ' ', cleanText)
        cleanText = re.sub('#\S+\s', ' ', cleanText)
        cleanText = re.sub('@\S+', '  ', cleanText)  
        cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
        cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
        cleanText = re.sub('\s+', ' ', cleanText)
        return cleanText
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
except Exception as e:
    st.error(f"An error occurred while loading models: {e}")


# web app

def main():
    st.title("resume screening app")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf',])
    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except:
            # if utf-8 decoding fails, try decoding with "latin-1"
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = cleanResume(resume_text)
        input_features = tfidfd.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]
        st.write(prediction_id)

        category_mapping = {
            15: "java dev",
            23: "testing", 
            8: "devops eng",
            20: "python dev",
            24:"web designing",
            12:"hr",
            13:"hadoop",
            3:"blockchain",
            10: "etl dev",
            18: "operations eng",
            6: "data science",
            22: "sales",
            16: "mech eng",
            1: "arts",
            7 : "database",
            11: "electrical eng",
            14: "health and fitness",
            19:"pwo",
            4:"business analyst",
            9:"dotnet dev",
            2 : "automation testing",
            17: "network security eng",
            21: "sap dev",
            5: "civil eng",
            0 : "advocate"
        }

        category_name = category_mapping.get(prediction_id, "Unknown")
        print("Predicted category : ", category_name)
        # print(prediction_id)
        

# python ka main
if __name__ == "__main__":
    main()