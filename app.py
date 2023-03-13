from flask import Flask, render_template,request
import pickle
import string
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
from nltk.stem import SnowballStemmer

ss = SnowballStemmer('english')

lst = ["no","nor","not","don","don't","ain","aren't","couldn","couldn't","didn","didn't","doesn","doesn't","hadn","hadn't","hasn","hasn't",
       "haven","haven't","isn","isn't","mightn","mightn't","mustn","mustn't","needn","needn't","shan","shan't","shouldn","shouldn't",
       "wasn","wasn't","weren","weren't","won","won't","wouldn","wouldn't"]
new_lst = []
for i in stopwords.words('english'):
    if i not in lst:
        new_lst.append(i)

model = pickle.load(open('model12.pkl','rb'))
vectorizer = pickle.load(open('vectorizer12.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/book1', methods=['POST'])
def home():
        
    def transform_text(text):
        text = text.lower()
        text = nltk.word_tokenize(text)
    
        y = []
    
        for i in text:
            if i.isalnum():
                y.append(i)
        b = []      
        for i in text:
            if i not in new_lst and i not in string.punctuation:
                b.append(i)
            
        c = []
        for i in b:
            c.append(ss.stem(i))
            
        return " ".join(c)
    
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        tranformed_msg = transform_text(user_input)
        vector_input = vectorizer.transform([tranformed_msg])
        result = model.predict(vector_input)[0]
        if result == 1:
            print("Positive Review")
            return render_template('index.html', result = result)
        else:
            print("Negative Review")
            return render_template('index.html', result = result)
    

if __name__=='__main__':
    app.run(debug=True)